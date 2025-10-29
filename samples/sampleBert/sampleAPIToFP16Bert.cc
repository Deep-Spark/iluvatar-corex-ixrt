#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "NvInfer.h"
#include "NvInferImpl.h"
#include "NvInferPlugin.h"
#include "NvInferRuntime.h"
#include "NvInferRuntimeCommon.h"
#include "NvInferRuntimePlugin.h"
#include "cuda_fp16.h"
#include "driver_types.h"
#include "json.hpp"
#include "logging.h"
#include "memory_utils.h"
#include "misc.h"
using json = nlohmann::json;
using namespace std;
using namespace nvinfer1;

#define WQKV "self_qkv_kernel"
#define BQKV "self_qkv_bias"
#define W_AOUT "attention_output_dense_kernel"
#define B_AOUT "attention_output_dense_bias"
#define W_MID "intermediate_dense_kernel"
#define B_MID "intermediate_dense_bias"
#define SQD_W "squad_output_weights"
#define SQD_B "squad_output_bias"
#define W_LOUT "output_dense_kernel"
#define B_LOUT "output_dense_bias"

namespace nvinfer1::samples {
using namespace common;
class TensorRTBertFP16Sample {
   public:
    bool build(string weight_path, string engine_path);
    bool infer(vector<vector<int32_t>>& input);
    void loadModelWeights(const std::string weight_path);
    bool buildNetwork();
    bool buildEngine();
    ILayer* addEmbLayerNorm();
    ITensor* addBertModel(ITensor* input, ITensor* input_mask);
    ILayer* addSquadOut(string prefix, ITensor* input);
    ILayer* addTransformerLayer(string prefix, ITensor* input, ITensor* input_mask);
    ILayer* addAtteionLayer(string prefix, ITensor* input, ITensor* input_mask);
    ILayer* addCustomFc(string prefix, ITensor* input, Weights& W, Weights* B, int output_dim);
    ILayer* addSkipLayernorm(string prefix, ITensor* input, ITensor* skip, Weights& bias);
    ILayer* addFFN(string prefix, ITensor* input);
    void getBertConfig(string config_file, int batch_size, int seq_len);
    void setOutputName(ITensor* x, string name);
    void setEngineSavePath(string path) { engine_save_path_ = path; }
    void debugEngine();
    void debugNetwork();
    void setBatchSize(int batch_size) { batch_size_ = batch_size; }
    int32_t getBatchSize() const { return batch_size_; }
    void setSeqLen(int seq_len) { seq_len_ = seq_len; }
    int32_t getSeqLen() const { return seq_len_; }

   private:
    Logger logger_;
    string engine_save_path_;
    UPtr<IBuilder> builder_;
    UPtr<INetworkDefinition> network_;
    UPtr<IBuilderConfig> config_;
    UPtr<IHostMemory> plan_;
    UPtr<ICudaEngine> engine_;
    UPtr<IRuntime> runtime_;
    UPtr<nvinfer1::IExecutionContext> context_;
    int32_t batch_size_;
    int32_t seq_len_;
    int32_t hidden_size_;
    int32_t num_layers_;
    int32_t num_head_;
    int32_t intermediate_size_;
    int32_t vocab_size_;
    int32_t max_position_embeddings_;
    int32_t type_vocab_size_;
    map<string, Weights> weight_map_;
    vector<string> input_names;
    vector<string> output_names;
    std::vector<nvinfer1::IPluginV2*> plugins_;
};

inline void DumpBuffer2Disk(const std::string& file_path, void* data, uint64_t len) {
    std::ofstream out_file(file_path, std::ios::binary);
    if (not out_file.is_open()) {
        out_file.close();
        return;
    }
    out_file.write((char*)data, len);
    out_file.close();
    cout << "Dump buffer size " << len << endl;
}

inline void LoadBufferFromDisk(const std::string& file_path, std::vector<int8_t>* engine_buffer) {
    std::ifstream in_file(file_path, std::ios::binary);
    if (not in_file.is_open()) {
        in_file.close();
        return;
    }
    in_file.seekg(0, std::ios::end);
    uint64_t file_length = in_file.tellg();
    in_file.seekg(0, std::ios::beg);
    engine_buffer->resize(file_length);
    in_file.read((char*)engine_buffer->data(), file_length);
    in_file.close();
    cout << "Load buffer size " << file_length << endl;
}

void TensorRTBertFP16Sample::loadModelWeights(const string weight_path) {
    std::cout << "Loading weights: " << weight_path << std::endl;

    // Open weights file
    ifstream input(weight_path);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    std::cout << count << std::endl;
    assert(count > 0 && "Invalid weight map file.");

    while (count--) {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = nvinfer1::DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (auto x = 0; x < size; x++) {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        wt.count = size;
        weight_map_[name] = wt;
    }
    std::cout << "Loading weights successfully!" << std::endl;
}

ILayer* TensorRTBertFP16Sample::addEmbLayerNorm() {
    auto input_ids = network_->addInput("input_ids", DataType::kINT32, Dims{2, {batch_size_, seq_len_}});
    auto segment_ids = network_->addInput("segment_ids", DataType::kINT32, Dims{2, {batch_size_, seq_len_}});
    auto input_mask = network_->addInput("input_mask", DataType::kINT32, Dims{2, {batch_size_, seq_len_}});

    auto emln_plg_creator = getPluginRegistry()->getPluginCreator("CustomEmbLayerNormPluginDynamic_IxRT", "1");
    vector<PluginField> pf;
    pf.emplace_back(PluginField("bert_embeddings_layernorm_beta", weight_map_["bert_embeddings_layernorm_beta"].values,
                                PluginFieldType::kFLOAT32,
                                (int32_t)weight_map_["bert_embeddings_layernorm_beta"].count));
    pf.emplace_back(PluginField("bert_embeddings_layernorm_gamma",
                                weight_map_["bert_embeddings_layernorm_gamma"].values, PluginFieldType::kFLOAT32,
                                (int32_t)weight_map_["bert_embeddings_layernorm_gamma"].count));
    pf.emplace_back(PluginField("bert_embeddings_word_embeddings",
                                weight_map_["bert_embeddings_word_embeddings"].values, PluginFieldType::kFLOAT32,
                                (int32_t)weight_map_["bert_embeddings_word_embeddings"].count));
    pf.emplace_back(PluginField("bert_embeddings_token_type_embeddings",
                                weight_map_["bert_embeddings_token_type_embeddings"].values, PluginFieldType::kFLOAT32,
                                (int32_t)weight_map_["bert_embeddings_token_type_embeddings"].count));
    pf.emplace_back(PluginField("bert_embeddings_position_embeddings",
                                weight_map_["bert_embeddings_position_embeddings"].values, PluginFieldType::kFLOAT32,
                                (int32_t)weight_map_["bert_embeddings_position_embeddings"].count));

    int output_fp16 = 1;
    int mha_type_id = 1;  // 3 FP32 2 FP16 1 INT8
    // int pad_id = 0;
    pf.emplace_back(PluginField("output_fp16", &output_fp16, PluginFieldType::kINT32));
    pf.emplace_back(PluginField("mha_type_id", &mha_type_id, PluginFieldType::kINT32));
    // diff
    // pf.emplace_back(PluginField("pad_id", &pad_id, PluginFieldType::kINT32));

    const PluginFieldCollection* plugin_collection =
        new PluginFieldCollection{static_cast<int32_t>(pf.size()), pf.data()};
    auto emln_plg = emln_plg_creator->createPlugin("embeddings", plugin_collection);
    plugins_.push_back(emln_plg);
    vector<ITensor*> emln_inputs = {input_ids, segment_ids, input_mask};
    auto emb_layer = network_->addPluginV2(emln_inputs.data(), emln_inputs.size(), *emln_plg);
    // diff no shuffle
    return emb_layer;
}

ILayer* TensorRTBertFP16Sample::addAtteionLayer(string prefix, ITensor* input, ITensor* input_mask) {
    // B*S*E*1*1 3E*E = B*S*(3E)*1*1
    // auto mult_all = network_->addFullyConnected(*input, 3 * hidden_size_, weight_map_[prefix + WQKV],
    // weight_map_[prefix + BQKV]);
    auto mult_all =
        addCustomFc(prefix, input, weight_map_[prefix + WQKV], &weight_map_[prefix + BQKV], 3 * hidden_size_);
    auto qkv2_plg_creator = getPluginRegistry()->getPluginCreator("CustomQKVToContextPluginDynamic_IxRT", "1");
    vector<PluginField> pf;
    int type_id = 1;
    int has_mask = input_mask == nullptr ? 0 : 1;
    pf.emplace_back(PluginField("type_id", &type_id, PluginFieldType::kINT32));
    pf.emplace_back(PluginField("hidden_size", &hidden_size_, PluginFieldType::kINT32));
    pf.emplace_back(PluginField("num_heads", &num_head_, PluginFieldType::kINT32));
    pf.emplace_back(PluginField("has_mask", &has_mask, PluginFieldType::kINT32));

    const PluginFieldCollection* plugin_collection =
        new PluginFieldCollection{static_cast<int32_t>(pf.size()), pf.data()};
    // B*S*(3*E)*1*1 -> B*S*(E)*1*1
    auto qkv2ctx_plg = qkv2_plg_creator->createPlugin("qkv2ctx", plugin_collection);
    plugins_.push_back(qkv2ctx_plg);
    vector<ITensor*> qkv2ctx_inputs = {mult_all->getOutput(0)};
    if (has_mask == 1) {
        qkv2ctx_inputs.emplace_back(input_mask);
    }
    auto qkv2ctx_layer = network_->addPluginV2(qkv2ctx_inputs.data(), qkv2ctx_inputs.size(), *qkv2ctx_plg);
    return qkv2ctx_layer;
}

ILayer* TensorRTBertFP16Sample::addCustomFc(string prefix, ITensor* input, Weights& W, Weights* B, int output_dim) {
    auto fc_creator = getPluginRegistry()->getPluginCreator("CustomFCPluginDynamic_IxRT", "1");
    vector<PluginField> pf;
    int type_id = 1;  // 3 FP32 2 FP16 1 INT8
    pf.emplace_back(PluginField("out_dims", &output_dim, PluginFieldType::kINT32));
    pf.emplace_back(PluginField("W", W.values, PluginFieldType::kFLOAT32, (int32_t)W.count));
    if (B != nullptr) pf.emplace_back(PluginField("B", B->values, PluginFieldType::kFLOAT32, (int32_t)B->count));
    pf.emplace_back(PluginField("type_id", &type_id, PluginFieldType::kINT32));
    const PluginFieldCollection* plugin_collection =
        new PluginFieldCollection{static_cast<int32_t>(pf.size()), pf.data()};
    auto fc_plg = fc_creator->createPlugin("fcplugin", plugin_collection);
    plugins_.push_back(fc_plg);
    vector<ITensor*> fc_inputs = {input};
    auto fc_layer = network_->addPluginV2(fc_inputs.data(), fc_inputs.size(), *fc_plg);
    return fc_layer;
}

ILayer* TensorRTBertFP16Sample::addSkipLayernorm(string prefix, ITensor* input, ITensor* skip, Weights& bias) {
    auto wbeta = weight_map_[prefix + "beta"];
    auto wgamma = weight_map_[prefix + "gamma"];
    auto skipln_creator = getPluginRegistry()->getPluginCreator("CustomSkipLayerNormPluginDynamic_IxRT", "1");
    vector<PluginField> pf;
    int type_id = 1;
    pf.emplace_back(PluginField("ld", &hidden_size_, PluginFieldType::kINT32));
    pf.emplace_back(PluginField("beta", wbeta.values, PluginFieldType::kFLOAT32, wbeta.count));
    pf.emplace_back(PluginField("gamma", wgamma.values, PluginFieldType::kFLOAT32, wgamma.count));
    pf.emplace_back(PluginField("type_id", &type_id, PluginFieldType::kINT32));
    if (bias.count != 0) {
        pf.emplace_back(PluginField("bias", bias.values, PluginFieldType::kFLOAT32, (int32_t)bias.count));
    }

    const PluginFieldCollection* plugin_collection =
        new PluginFieldCollection{static_cast<int32_t>(pf.size()), pf.data()};
    auto skipln_plg = skipln_creator->createPlugin("skipln", plugin_collection);
    plugins_.push_back(skipln_plg);
    vector<ITensor*> skipln_inputs = {input, skip};
    auto skipln_layer = network_->addPluginV2(skipln_inputs.data(), skipln_inputs.size(), *skipln_plg);
    return skipln_layer;
}

ILayer* TensorRTBertFP16Sample::addFFN(string prefix, ITensor* input) {
    auto B_mid = weight_map_[prefix + B_MID];
    auto W_mid = weight_map_[prefix + W_MID];
    // [B, S, E] * [i, E] = [B, S, i]
    // auto mid_dense = network_->addFullyConnected(*input, intermediate_size_, W_mid, B_mid);
    auto mid_dense = addCustomFc(prefix + "mid_dense", input, W_mid, &B_mid, intermediate_size_);
    auto mid_dense_out = mid_dense->getOutput(0);

    // gelu
    auto gelu_creator = getPluginRegistry()->getPluginCreator("CustomGeluPluginDynamic_IxRT", "1");
    vector<PluginField> pf;
    int type_id = 1;  // 3 FP32 2 FP16 1 INT8
    pf.emplace_back(PluginField("type_id", &type_id, PluginFieldType::kINT32));
    pf.emplace_back(PluginField("ld", &intermediate_size_, PluginFieldType::kINT32));
    const PluginFieldCollection* plugin_collection =
        new PluginFieldCollection{static_cast<int32_t>(pf.size()), pf.data()};
    auto gelu_plg = gelu_creator->createPlugin("gelu", plugin_collection);
    plugins_.push_back(gelu_plg);
    vector<ITensor*> gelu_inputs = {mid_dense_out};
    auto gelu_layer = network_->addPluginV2(gelu_inputs.data(), gelu_inputs.size(), *gelu_plg);
    auto W_loutT = weight_map_[prefix + W_LOUT];
    auto B_loutT = weight_map_[prefix + B_LOUT];
    auto out_dense_layer =
        addCustomFc(prefix + "mid_dense_out", gelu_layer->getOutput(0), W_loutT, nullptr, hidden_size_);
    auto out_skip_layer = addSkipLayernorm(prefix + "output_layernorm_", out_dense_layer->getOutput(0), input, B_loutT);
    return out_skip_layer;
}

ILayer* TensorRTBertFP16Sample::addTransformerLayer(string prefix, ITensor* input, ITensor* input_mask) {
    auto qkv2ctx_layer = addAtteionLayer(prefix + "attention_", input, input_mask);
    // [B, S, iE, 1, 1] * [oE, iE]
    auto W_aoutT = weight_map_[prefix + W_AOUT];
    // auto qkv_fc_layer = network_->addFullyConnected(*qkv2ctx_layer->getOutput(0), hidden_size_, W_aoutT, {});
    auto qkv_fc_layer = addCustomFc(prefix, qkv2ctx_layer->getOutput(0), W_aoutT, nullptr, hidden_size_);
    auto B_aout = weight_map_[prefix + B_AOUT];
    auto skip_layer =
        addSkipLayernorm(prefix + "attention_output_layernorm_", qkv_fc_layer->getOutput(0), input, B_aout);

    auto ffn_layer = addFFN(prefix, skip_layer->getOutput(0));

    return ffn_layer;
}

ITensor* TensorRTBertFP16Sample::addBertModel(ITensor* input, ITensor* input_mask) {
    ITensor* pre_input = input;
    for (auto layer_id_ = 0; layer_id_ < num_layers_; layer_id_++) {
        auto prefix = "l" + to_string(layer_id_) + "_";
        auto out_layer = addTransformerLayer(prefix, pre_input, input_mask);
        pre_input = out_layer->getOutput(0);
    }
    return pre_input;
}

void TensorRTBertFP16Sample::setOutputName(ITensor* x, string name) { x->setName(name.c_str()); }

ILayer* TensorRTBertFP16Sample::addSquadOut(string prefix, ITensor* input) {
    auto W_out = weight_map_[prefix + SQD_W];
    auto B_out = weight_map_[prefix + SQD_B];
    auto dense = addCustomFc(prefix, input, W_out, &B_out, 2);
    // auto dense = network_->addFullyConnected(*input, 2, W_out, B_out);
    // diff not shuffle
    // setOutputName(dense->getOutput(0), prefix + "squad_logits");
    return dense;
}

bool TensorRTBertFP16Sample::buildNetwork() {
    auto emb_layer = addEmbLayerNorm();
    auto embeddings = emb_layer->getOutput(0);
    auto mask_idx = emb_layer->getOutput(1);

    // diff
    auto bert_out = addBertModel(embeddings, mask_idx);
    // setOutputName(bert_out, "bert_out");
    auto squad_logits = addSquadOut("cls_", bert_out);
    auto squad_logits_out = squad_logits->getOutput(0);
    // printf("add_squad_logits..\n");
    squad_logits->setOutputType(0, DataType::kFLOAT);
    network_->markOutput(*squad_logits_out);
    return true;
}

bool TensorRTBertFP16Sample::buildEngine() {
    plan_ = UPtr<nvinfer1::IHostMemory>(builder_->buildSerializedNetwork(*network_, *config_));
    if (!plan_) {
        std::cout << "Create serialized engine plan failed" << std::endl;
        return false;
    } else {
        std::cout << "Create serialized engine plan successfully!" << std::endl;
    }
    //
    if (engine_save_path_ != "") {
        DumpBuffer2Disk(engine_save_path_, plan_->data(), plan_->size());
    }
    return true;
}

void TensorRTBertFP16Sample::debugNetwork() {
    cout << "###########################################" << endl;
    cout << "layer number: " << network_->getNbLayers() << endl;
    auto num_input = network_->getNbInputs();
    cout << "number of input: " << num_input << endl;
    auto num_output = network_->getNbOutputs();
    cout << "number of output: " << num_output << endl;

    for (auto i = 0; i < num_input; i++) {
        auto in = network_->getInput(i);
        nvinfer1::Dims inputDims = network_->getInput(i)->getDimensions();
        cout << "\nInput " << i << ", " << network_->getInput(i)->getName() << " dims: " << endl;
        for (auto j = 0; j < inputDims.nbDims; ++j) {
            cout << inputDims.d[j] << " ";
        }
        cout << endl;
    }
    for (auto i = 0; i < num_output; i++) {
        nvinfer1::Dims outputDims = network_->getOutput(i)->getDimensions();
        cout << "\nOutput " << i << ", " << network_->getOutput(i)->getName() << " dims: " << endl;
        for (auto j = 0; j < outputDims.nbDims; ++j) {
            cout << outputDims.d[j] << " ";
        }
        cout << endl;
    }
    cout << "###########################################" << endl;
}

bool TensorRTBertFP16Sample::build(string weight_path, string engine_path) {
    loadModelWeights(weight_path);
    setEngineSavePath(engine_path);
    initLibNvInferPlugins(&logger_, "");
    logger_ = Logger(nvinfer1::ILogger::Severity::kINFO);
    // build
    builder_ = UPtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
    if (!builder_) {
        std::cout << "Create builder failed" << std::endl;
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    network_ = UPtr<nvinfer1::INetworkDefinition>(builder_->createNetworkV2(explicitBatch));
    if (!network_) {
        std::cout << "Create network failed" << std::endl;
        return false;
    }

    config_ = UPtr<nvinfer1::IBuilderConfig>(builder_->createBuilderConfig());
    if (!config_) {
        std::cout << "Create config failed" << std::endl;
        return false;
    }
    config_->setFlag(nvinfer1::BuilderFlag::kFP16);

    if (!buildNetwork()) {
        std::cout << "buidNetwork failed!!!" << std::endl;
        return false;
    } else {
        std::cout << "buidNetwork success!!!" << std::endl;
    }

    debugNetwork();

    if (!buildEngine()) {
        std::cout << "buildEngine failed!!!" << std::endl;
        return false;
    } else {
        std::cout << "buildEngine success!!!" << std::endl;
    }

    for (auto p : plugins_) {
        if (p) {
            p->destroy();
        }
    }
    return true;
}

void TensorRTBertFP16Sample::debugEngine() {
    cout << "###########################################" << endl;
    cout << "Engine name: " << engine_->getName() << endl;
    auto num_bd = engine_->getNbBindings();
    cout << "Number of binding data: " << num_bd << endl;
    for (auto i = 0; i < num_bd; ++i) {
        cout << "The " << i << " binding" << endl;
        cout << "Name: " << engine_->getBindingName(i) << endl;
        cout << "Format: " << (int32_t)engine_->getBindingFormat(i) << endl;
        cout << "Data type: " << (int32_t)engine_->getBindingDataType(i) << endl;
        cout << "Dimension: ";
        for (auto k = 0; k < engine_->getBindingDimensions(i).nbDims; ++k) {
            cout << engine_->getBindingDimensions(i).d[k] << " ";
        }
        cout << endl;
    }
    for (auto i = 0; i < input_names.size(); i++) {
        cout << "Input index " << i << ": " << engine_->getBindingIndex(input_names[i].c_str()) << endl;
    }
    for (auto i = 0; i < output_names.size(); i++) {
        cout << "Output index " << i + (int)input_names.size() << ": "
             << engine_->getBindingIndex(output_names[i].c_str()) << endl;
    }
    cout << "###########################################" << endl;
}

bool TensorRTBertFP16Sample::infer(vector<vector<int32_t>>& input) {
    runtime_ = UPtr<IRuntime>(createInferRuntime(logger_));
    if (!engine_save_path_.empty()) {
        std::vector<int8_t> engine_buffer;
        LoadBufferFromDisk(engine_save_path_, &engine_buffer);
        engine_ = UPtr<ICudaEngine>(runtime_->deserializeCudaEngine(engine_buffer.data(), engine_buffer.size()));
    } else {
        engine_ = UPtr<ICudaEngine>(runtime_->deserializeCudaEngine(plan_->data(), plan_->size()));
    }

    if (!engine_) {
        std::cout << "Create engine failed" << std::endl;
        return false;
    } else {
        std::cout << "Create engine done" << endl;
    }

    // set binding buffer
    std::vector<void*> binding_buffer(engine_->getNbBindings());

    void* input_ids_gpu{nullptr};
    auto input_size = batch_size_ * seq_len_ * sizeof(int32_t);
    CHECK(cudaMalloc(&input_ids_gpu, input_size));
    CHECK(cudaMemcpy(input_ids_gpu, input[0].data(), input_size, cudaMemcpyHostToDevice));
    auto input_ids_index = engine_->getBindingIndex("input_ids");
    binding_buffer.at(input_ids_index) = input_ids_gpu;

    void* segment_ids_gpu{nullptr};
    CHECK(cudaMalloc(&segment_ids_gpu, input_size));
    CHECK(cudaMemcpy(segment_ids_gpu, input[1].data(), input_size, cudaMemcpyHostToDevice));
    auto segment_ids_index = engine_->getBindingIndex("segment_ids");
    binding_buffer.at(segment_ids_index) = segment_ids_gpu;

    void* input_mask_gpu{nullptr};
    CHECK(cudaMalloc(&input_mask_gpu, input_size));
    CHECK(cudaMemcpy(input_mask_gpu, input[2].data(), input_size, cudaMemcpyHostToDevice));
    auto input_mask_index = engine_->getBindingIndex("input_mask");
    binding_buffer.at(input_mask_index) = input_mask_gpu;

    auto output_index = engine_->getBindingIndex(network_->getOutput(0)->getName());
    void* output_gpu{nullptr};
    CHECK(cudaMalloc(&output_gpu, volume(network_->getOutput(0)->getDimensions()) * sizeof(float)));
    binding_buffer.at(output_index) = output_gpu;

    debugEngine();

    // execute context
    context_ = UPtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context_) {
        cout << "Create execution context failed" << endl;
        return false;
    } else {
        cout << "Create execution context successfully!" << endl;
    }

    // Warmup
    for (auto i = 0; i < 1; ++i) {
        context_->executeV2(binding_buffer.data());
    }

    auto start = NowUs();
    auto status = context_->executeV2(binding_buffer.data());
    uint64_t time = NowUs() - start;
    if (!status) {
        cout << "Execute tensorrt failed" << endl;
        return false;
    }

    float fps = 1 / ((float)time / 1000000);
    std::cout << "BatchSize: " << batch_size_ << ", SeqLen: " << seq_len_ << ", FPS: " << fps << std::endl;

    int num = batch_size_ * seq_len_ * 2;
    float* cpu_output = (float*)malloc(num * sizeof(float));
    CHECK(cudaMemcpy(cpu_output, output_gpu, num * sizeof(float), cudaMemcpyDeviceToHost));
    for (auto i = 0; i < num; i++) {
        printf("%f\n", (cpu_output[i]));
    }
    printf("\n");
    // printf("%f %f\n", __half2float(cpu_output[0]), __half2float(cpu_output[1]));
    return true;
}

void TensorRTBertFP16Sample::getBertConfig(string config_file, int batch_size, int seq_len) {
    setBatchSize(batch_size);
    setSeqLen(seq_len);
    std::ifstream in(config_file);
    json bert;
    in >> bert;
    hidden_size_ = bert["hidden_size"];
    num_layers_ = bert["num_hidden_layers"];
    num_head_ = bert["num_attention_heads"];
    intermediate_size_ = bert["intermediate_size"];

    cout << "Config info:" << endl;
    cout << "batch_size_ = " << batch_size_ << endl;
    cout << "seq_len_ = " << seq_len_ << endl;
    cout << "hidden_size_ = " << hidden_size_ << endl;
    cout << "num_layers_ = " << num_layers_ << endl;
    cout << "num_head_ = " << num_head_ << endl;
    cout << "intermediate_size_ = " << intermediate_size_ << endl;
}
}  // namespace nvinfer1::samples

int main() {
    using namespace nvinfer1::samples;
    vector<vector<int32_t>> input = {
        {// input_ids
         101,   2054,  2003,  23435, 5339,  1029, 102,   23435, 5339,  2003,  1037,  2152,  2836,  2784,  4083,  28937,
         4132,  2008,  18058, 2659,  2397,  9407, 1998,  2152,  2083,  18780, 2005,  18726, 2107,  2004,  16755, 2545,
         1010,  4613,  1998,  3746,  1013,  2678, 2006,  1050,  17258, 2401,  14246, 2271,  1012,  2009,  2950,  11968,
         8043,  2015,  2000,  12324, 4275,  1010, 1998,  13354, 7076,  2000,  2490,  3117,  23092, 1998,  9014,  2077,
         11243, 20600, 2015,  2005,  28937, 1012, 2651,  1050,  17258, 2401,  2003,  2330,  1011,  14768, 6129,  11968,
         8043,  2015,  1998,  13354, 7076,  1999, 23435, 5339,  2061,  2008,  1996,  2784,  4083,  2451,  2064,  7661,
         4697,  1998,  7949,  2122,  6177,  2000, 2202,  5056,  1997,  3928,  23435, 5339,  20600, 2015,  2005,  2115,
         18726, 1012,  102,   0,     0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
         0,     0,     0,     0,     0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
         0,     0,     0,     0,     0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
         0,     0,     0,     0,     0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
         0,     0,     0,     0,     0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
         0,     0,     0,     0,     0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
         0,     0,     0,     0,     0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
         0,     0,     0,     0,     0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
         0,     0,     0,     0,     0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
         0,     0,     0,     0,     0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
         0,     0,     0,     0,     0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
         0,     0,     0,     0,     0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
         0,     0,     0,     0,     0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
         0,     0,     0,     0,     0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
         0,     0,     0,     0,     0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
         0,     0,     0,     0,     0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
         0,     0,     0,     0,     0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,     0},
        {// segment_ids
         0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {// input_mask
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
    TensorRTBertFP16Sample sample;
    auto weight_path = "../../oss/samples/sampleBert/data/bert-large-uncased/bert_large_v1_1.wts";
    auto config_file = "../../oss/samples/sampleBert/data/bert-large-uncased/bert_config.json";
    auto engine_path = "../../oss/samples/sampleBert/data/bert_large_384.engine";
    sample.getBertConfig(config_file, 1, 384);
    sample.build(weight_path, "");
    sample.infer(input);
    return 0;
}