def apply(
        self,
        layer: torch.nn.Module,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        bias: Optional[torch.Tensor] = None,
        x_transform: Optional[str] = None,
        x_dim: Optional[int] = 0,
        throw_dequant: Optional[bool] = False,
    ) -> torch.Tensor:
        if isinstance(x, Dict):
            x_scale = x.get('pertoken_scale', None)
            x = x.get('x_int8', None)
        else:
            x, x_scale = torch_npu.npu_dynamic_quant(x)
        # TODO scale_parallel is not supported yet. scale_parallel = model_extra_config.operator_opt_config.enable_scale_parallel
        if x_transform == "AllGather":
            x_scale = layer_parallel_all_gather(
                x_scale, layer.layer_name_inside_block, "x", x_dim
            )
            x = layer_parallel_all_gather(x, layer.layer_name_inside_block, "x", x_dim)
        elif x_transform == "ALL2ALL":
            x_scale = layer_parallel_all2all_single(
                x_scale, layer.layer_name_inside_block, "x", x_dim
            )
            x = layer_parallel_all2all_single(
                x, layer.layer_name_inside_block, "x", x_dim
            )
        if throw_dequant and bias is None:
            if torch.distributed.get_rank() == 1:
                print(f"[TEST] w8a8 fc 1 before matmul: {x[-127:].double().sum()=:.10e} {x.shape=} {layer.weight.double().sum()=:.10e} {layer.weight.shape=} {layer.weight_scale=}")
            y = torch_npu.npu_quant_matmul(
                x1=x,
                x2=layer.weight,
                scale=layer.weight_scale,
                bias=None,
                output_dtype=torch.int32,
            )
            if torch.distributed.get_rank() == 1:
                print(f"[TEST] w8a8 fc 1 after matmul: {y[-127:].double().sum()=:.10e} {y.shape=}")
            return y, x_scale
        else:
            if torch.distributed.get_rank() == 1:
                print(f"[TEST] w8a8 fc 2 before matmul: {x[-127:].double().sum()=:.10e} {x.shape=} {layer.weight.double().sum()=:.10e} {layer.weight.shape=} {layer.weight_scale=} {layer.orig_dtype=}")
                if bias is not None:
                    print(f"[TEST] w8a8 fc 2 before matmul: {bias.double().sum()=:.10e} {bias.shape=}")

            y = torch_npu.npu_quant_matmul(
                x1=x,
                x2=layer.weight,
                scale=layer.weight_scale,
                pertoken_scale=x_scale,
                bias=bias,
                output_dtype=layer.orig_dtype,
            )
            if torch.distributed.get_rank() == 1:
                print(f"[TEST] w8a8 fc 2 after matmul: {y[-127:].double().sum()=:.10e} {y.shape=}")
            return y