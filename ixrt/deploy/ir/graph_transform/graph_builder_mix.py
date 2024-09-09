# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
#

from typing import List, Union

from .. import utils
from ..operator import Operator
from ..operator_attr import DynamicAttr
from ..variable import Placeholder, Variable, VariableOptions
from .base_transform import BaseGraphTransform


class GraphBuilderMix(BaseGraphTransform):
    def _add_variable_if_not_exist(self, var: Union[str, Variable]) -> Variable:
        if self.graph.containe_var(var):
            if isinstance(var, str):
                var = self.graph.get_variable(var)
            return var
        else:
            if isinstance(var, str):
                var = Placeholder(var)
            self.graph.add_variable(var)
        return var

    def make_operator(
        self,
        op_type: str,
        name: str = None,
        inputs: Union[
            str, Variable, Operator, List[Union[str, Variable, Operator]]
        ] = None,
        outputs: Union[str, Variable, List[Union[str, Variable]]] = None,
        attr_cls=None,
        **attrs,
    ) -> Operator:
        """
        创建算子并增加到图中

        Example:
            >>> op = transform.make_operator(
            >>>     "Add",
            >>>     name="Add1",
            >>>     inputs=["a", "b"],
            >>>     outputs=["c"]
            >>> )

        :param op_type: 算子类型
        :param name: 算子名字，如果为 None，那么会自动生成名字
        :param inputs: 算子的输入
        :param outputs: 算子的输出
        :param attr_cls: 算子属性的类
        :param attrs: 算子的属性
        :return: Operator
        """
        if name is None:
            name = utils.generate_operator_name(
                self.graph, pattern=str(op_type) + "_{idx}"
            )

        if inputs is None:
            inputs = []
        elif isinstance(inputs, (Operator, str, Variable)):
            inputs = [inputs]

        if outputs is None:
            outputs = []
        elif isinstance(outputs, (str, Variable)):
            outputs = [outputs]

        new_inputs = []
        for input in inputs:
            if isinstance(input, Operator):
                for parent_op_out in input.outputs:
                    parent_op_out = self._add_variable_if_not_exist(parent_op_out).name
                    new_inputs.append(parent_op_out)
            else:
                input = self._add_variable_if_not_exist(input).name
                new_inputs.append(input)

        new_outputs = []
        for out in outputs:
            out = self._add_variable_if_not_exist(out).name
            new_outputs.append(out)

        if attr_cls is None:
            attr = DynamicAttr(attrs)
        else:
            attr = attr_cls(**attrs)

        op = Operator(
            name=name,
            op_type=op_type,
            inputs=new_inputs,
            outputs=new_outputs,
            attributes=attr,
        )
        self.graph.add_operator(op)
        return op

    def make_variable(self, name=None, value=None, **options):
        """
        创建变量，并自动加入到图中

        Example:
            >>> var = transform.make_variable()
            >>> var1 = transform.make_variable(name="var1")
            >>> var2 = transform.make_variable(name="var2", value=torch.randn(1))

        :param name: 变量名，如果为空，将自动生成变量名
        :param value: 变量的值，可以为标量，可以为 PyTorch 的 Tesnor，也可以是 numpy 的数组
        :param options: 变量的属性描述，参考 VariableOptions，可以不传参数
        :return: Variable
        """
        if name is None:
            name = utils.generate_variable_name(self.graph)

        var = Variable(name=name, value=value, options=VariableOptions(**options))
        self.graph.add_variable(var)
        return var
