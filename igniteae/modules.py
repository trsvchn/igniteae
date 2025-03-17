from typing import Any, Callable, NamedTuple, Type
from functools import reduce
from operator import add
import math
from enum import Enum, member

import torch
from torch import distributions, nn, Tensor


class Id(nn.Identity):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.encoder = nn.Parameter(torch.empty(1,))
        self.decoder = nn.Parameter(torch.empty(1,))


class Affine(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        nonlinearity="leaky_relu",
    ) -> None:
        self.nonlinearity = nonlinearity
        super().__init__(in_features, out_features, bias, device, dtype)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5), nonlinearity=self.nonlinearity)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)


class KLDivLoss(nn.Module):
    def __init__(self, reduction="mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        p = input
        q = torch.distributions.Normal(torch.zeros_like(p.loc), torch.ones_like(p.scale))
        kld_loss = torch.distributions.kl.kl_divergence(p, q)
        if self.reduction == "sum":
            kld_loss = kld_loss.sum()
        elif self.reduction == "mean":
            kld_loss = kld_loss.mean()
        else:
            raise NotImplementedError
        return kld_loss


class Normal(nn.Module):
    def __init__(self, params_dim: int = -1):
        super().__init__()
        self.params_dim = params_dim
        self.encoder = distributions.Normal
        self.decoder = nn.Identity()

    def encode_distrib(self, input: Tensor) -> distributions.Normal:
        # input tensor has to be 3D
        loc, log_var = [_ for _ in input.split(split_size=1, dim=self.params_dim)]
        scale = torch.exp(0.5 * log_var)
        return self.encoder(loc, scale)

    def encode(self, input: Tensor) -> Tensor:
        return self.encode_distrib(input).rsample()

    def decode(self, input: Tensor) -> Tensor:
        return self.decoder(input)

    def forward(self, input: Tensor) -> Tensor:
        return self.decode(self.encode(input))


class Composer(nn.Sequential):
    def encode(self, input: Tensor) -> Tensor:
        for m in self:
            input = m.encode(input)
        return input

    def decode(self, input: Tensor) -> Tensor:
        for m in self[::-1]:
            input = m.decode(input)
        return input

    def forward(self, input):
        return self.decode(self.encode(input))


class IgnoredModule(nn.Module):
    def __init__(self, name, module):
        super().__init__()
        self.add_module(name, module)

    def forward(self, *args, **kwargs):
        return 0.0


class LossTerm(NamedTuple):
    name: str
    type: "LossType"
    weight: float | Callable[[], float] | Callable[[Any], float]
    module: nn.Module
    premodule: Callable[[Any, Any], Any]

    def unwrap_weight(self, state: Any | None = None) -> float:
        match self:
            case LossTerm(type=LossType.Static | LossType.Ignored, weight=value):
                return value  # type: ignore[return-value]
            case LossTerm(type=LossType.Dynamic, weight=value):
                return value() if state is None else value(state)  # type: ignore[call-arg, operator]
            case _:
                raise NotImplementedError


def create_loss_term(
    default_weight: float | None = None, ignored_module: Type[IgnoredModule] | None = None
) -> Callable:
    def init_loss_term(
        loss_type: "LossType",
        name: str,
        weight: float | Callable[[], float] | Callable[[Any], float],
        module: nn.Module,
        premodule: Callable[[Any, Any], Any],
    ) -> LossTerm:
        return LossTerm(
            name,
            loss_type,
            default_weight if default_weight is not None else weight,
            ignored_module(name, module) if ignored_module is not None else module,
            premodule,
        )

    return init_loss_term


class LossType(Enum):
    Static = member(create_loss_term())
    Dynamic = member(create_loss_term())
    Ignored = member(create_loss_term(0.0, IgnoredModule))

    def __call__(
        self,
        name: str,
        weight: float | Callable[[], float] | Callable[[Any], float],
        module: nn.Module,
        premodule: Callable[[Any, Any], Any],
    ) -> LossTerm:
        return self.value(self, name, weight, module, premodule)


class LossComposer(nn.ModuleDict):
    def __init__(self, loss_terms: list[LossTerm]) -> None:
        super().__init__({lt.name: lt.module for lt in loss_terms})
        self.loss_terms = {lt.name: lt for lt in loss_terms}

    def _parallel_forward(self, inputs, targets) -> dict[str, Tensor]:
        return {
            loss_term_name: loss_module(*self.loss_terms[loss_term_name].premodule(inputs, targets))
            for loss_term_name, loss_module in self.items()
        }

    def _apply_weights(self, loss_values: dict[str, Tensor], state=None) -> dict[str, Tensor]:
        return {
            loss_term_name: self.loss_terms[loss_term_name].unwrap_weight(state) * loss_value
            for loss_term_name, loss_value in loss_values.items()
        }

    def _reduce(self, weighted_loss_values: dict[str, Tensor]) -> Tensor:
        return reduce(add, weighted_loss_values.values(), 0.0)  # type: ignore[return-value]

    def forward(self, inputs, targets, state=None) -> Tensor:
        return self._reduce(self._apply_weights(self._parallel_forward(inputs, targets), state))


class LegacyLossComposer(nn.ModuleDict):
    def __init__(
        self,
        modules: dict[str, nn.Module] | None = None,
        weights: dict[str, float] | None = None,
        output_transforms: dict[str, Callable[[Any, Any], Any]] | None = None,
    ):
        super().__init__(modules=modules)
        self.weights = {}
        if weights is None:
            for k, v in self.items():
                self.weights[k] = 1.0
        else:
            for k, v in self.items():
                self.weights[k] = weights[k]

        if output_transforms is not None:
            for tk in output_transforms:
                if tk not in self:
                    raise ValueError(f"Loss function for `{tk}` output transform is not defined!")
            for mk in self:
                if mk not in output_transforms:
                    raise ValueError(f"Output transform for `{mk}` loss function is not defined!")
        self.output_transforms = output_transforms

    def forward(
        self,
        model_output: Tensor | dict[str, Tensor],
        batch: Tensor | dict[str, Tensor],
        weights: dict[str, float] | None = None,
    ):
        # Override self.weights with weights, Note: all of them!
        if weights is not None:
            # Check elements.
            for lk in weights:
                if lk not in self:
                    raise KeyError(f"Loss function for `{lk}` is not defined!")
            for mk in self:
                if mk not in weights:
                    raise KeyError(f"Weight for `{mk}` is not specified!")
        else:
            weights = self.weights
        loss = 0.0
        for k, loss_fn in self.items():
            k_model_output = model_output
            k_batch = batch
            weight = weights[k]
            # if weight == 0.0:
            #     continue
            if self.output_transforms is not None:
                k_model_output, k_batch = self.output_transforms[k](model_output, batch)
            loss += weight * loss_fn(k_model_output, k_batch)
        return loss
