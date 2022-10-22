from abc import ABC, abstractmethod

import numpy as np
import scipy.optimize as sopt
import torch


class BaseWrapper(ABC):
    def get_input(self, input_var):
        self.input_type = type(input_var)
        assert self.input_type in [
            dict,
            list,
            np.ndarray,
            torch.Tensor,
        ], "The initial input to your optimized function should be one of dict, list or np.ndarray"
        input_, self.shapes = self._concat(input_var)
        self.var_num = input_.shape[0]
        return input_

    def get_output(self, output_var):
        assert "shapes" in dir(self), "You must first call get input to define the tensors shapes."
        output_var_ = self._unconcat(output_var, self.shapes)
        return output_var_

    def get_bounds(self, bounds):

        if bounds is not None:
            if isinstance(bounds, tuple) and not (
                isinstance(bounds[0], tuple) or isinstance(bounds[0], sopt.Bounds)
            ):
                assert len(bounds) == 2
                new_bounds = [bounds] * self.var_num

            elif isinstance(bounds, sopt.Bounds):
                new_bounds = [bounds] * self.var_num

            elif type(bounds) in [list, tuple, np.ndarray]:
                if self.input_type in [list, tuple]:
                    assert len(self.shapes) == len(bounds)
                    new_bounds = []
                    for sh, bounds_ in zip(self.shapes, bounds):
                        new_bounds += format_bounds(bounds_, sh)
                elif self.input_type in [np.ndarray]:
                    new_bounds = bounds
                elif self.input_type in [torch.Tensor]:
                    new_bounds = bounds.detach().cpu().numpy()

            elif isinstance(bounds, dict):
                assert self.input_type == dict
                assert set(bounds.keys()).issubset(self.shapes.keys())

                new_bounds = []
                for k in self.shapes.keys():
                    if k in bounds.keys():
                        new_bounds += format_bounds(bounds[k], self.shapes[k])
                    else:
                        new_bounds += [(None, None)] ** np.prod(self.shapes[k], dtype=np.int32)
        else:
            new_bounds = bounds
        return new_bounds

    def get_constraints(self, constraints, method):
        if constraints is not None and not isinstance(constraints, sopt.LinearConstraint):
            assert isinstance(constraints, dict)
            assert "fun" in constraints.keys()
            self.ctr_func = constraints["fun"]
            use_autograd = constraints.get("use_autograd", True)
            if method in ["trust-constr"]:

                constraints = sopt.NonlinearConstraint(
                    self._eval_ctr_func,
                    lb=constraints.get("lb", -np.inf),
                    ub=constraints.get("ub", np.inf),
                    jac=self.get_ctr_jac if use_autograd else "2-point",
                    keep_feasible=constraints.get("keep_feasible", False),
                )
            elif method in ["COBYLA", "SLSQP"]:
                constraints = {
                    "type": constraints.get("type", "eq"),
                    "fun": self._eval_ctr_func,
                }
                if use_autograd:
                    constraints["jac"] = self.get_ctr_jac
            else:
                raise NotImplementedError
        elif constraints is None:
            constraints = ()
        return constraints

    @abstractmethod
    def get_value_and_grad(self, input_var):
        return

    @abstractmethod
    def get_hvp(self, input_var, vector):
        return

    @abstractmethod
    def get_hess(self, input_var):
        return

    def _eval_func(self, input_var):
        if isinstance(input_var, dict):
            loss = self.func(**input_var)
        elif isinstance(input_var, list) or isinstance(input_var, tuple):
            loss = self.func(*input_var)
        else:
            loss = self.func(input_var)
        return loss

    def _eval_ctr_func(self, input_var):
        input_var = self._unconcat(input_var, self.shapes)
        if isinstance(input_var, dict):
            ctr_val = self.ctr_func(**input_var)
        elif isinstance(input_var, list) or isinstance(input_var, tuple):
            ctr_val = self.ctr_func(*input_var)
        else:
            ctr_val = self.ctr_func(input_var)
        return ctr_val

    @abstractmethod
    def get_ctr_jac(self, input_var):
        return

    def _concat(self, ten_vals):
        ten = []
        if isinstance(ten_vals, dict):
            shapes = {}
            for k, t in ten_vals.items():
                if t is not None:
                    if isinstance(t, (np.floating, float, int)):
                        t = np.array(t)
                    shapes[k] = t.shape
                    ten.append(self._reshape(t, [-1]))
            ten = self._tconcat(ten, 0)

        elif isinstance(ten_vals, list) or isinstance(ten_vals, tuple):
            shapes = []
            for t in ten_vals:
                if t is not None:
                    if isinstance(t, (np.floating, float, int)):
                        t = np.array(t)
                    shapes.append(t.shape)
                    ten.append(self._reshape(t, [-1]))
            ten = self._tconcat(ten, 0)

        elif isinstance(ten_vals, (np.floating, float, int)):
            ten_vals = np.array(ten_vals)
            shapes = np.array(ten_vals).shape
            ten = self._reshape(np.array(ten_vals), [-1])
        elif isinstance(ten_vals, torch.Tensor):
            ten_vals = ten_vals.detach().cpu()
            shapes = ten_vals.shape
            ten = self._reshape(ten_vals, [-1])
        else:
            ten_vals = ten_vals
            shapes = ten_vals.shape
            ten = self._reshape(ten_vals, [-1])
        return ten, shapes

    def _unconcat(self, ten, shapes):
        current_ind = 0
        if isinstance(shapes, dict):
            ten_vals = {}
            for k, sh in shapes.items():
                next_ind = current_ind + np.prod(sh, dtype=np.int32)
                ten_vals[k] = self._reshape(self._gather(ten, current_ind, next_ind), sh)

                current_ind = next_ind

        elif isinstance(shapes, list) or isinstance(shapes, tuple):
            if isinstance(shapes[0], int):
                ten_vals = self._reshape(ten, shapes)
            else:
                ten_vals = []
                for sh in shapes:
                    next_ind = current_ind + np.prod(sh, dtype=np.int32)
                    ten_vals.append(self._reshape(self._gather(ten, current_ind, next_ind), sh))

                    current_ind = next_ind

        elif shapes is None:
            ten_vals = ten

        return ten_vals

    @abstractmethod
    def _reshape(self, t, sh):
        return

    @abstractmethod
    def _tconcat(self, t_list, dim=0):
        return

    @abstractmethod
    def _gather(self, t, i, j):
        return


def format_bounds(bounds_, sh):
    if isinstance(bounds_, tuple):
        assert len(bounds_) == 2
        return [bounds_] * np.prod(sh, dtype=np.int32)
    elif isinstance(bounds_, sopt.Bounds):
        return [bounds_] * np.prod(sh, dtype=np.int32)
    elif isinstance(bounds_, list):
        assert np.prod(sh) == len(bounds_)
        return bounds_
    elif isinstance(bounds_, np.ndarray):
        assert np.prod(sh) == np.prod(np.array(bounds_).shape)
        return np.concatenate(np.reshape(bounds_, -1)).tolist()
    else:
        raise TypeError
