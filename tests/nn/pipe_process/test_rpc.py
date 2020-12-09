import copy
import os

import pytest
import torch
from torch import nn
from torch.distributed import rpc

from fairscale.nn.model_parallel.initialize import get_pipeline_parallel_group
from fairscale.nn.pipe import PipeRPCWrapper
from fairscale.utils.testing import get_worker_map, torch_spawn


def init_rpc(offset=0):
    os.environ["MASTER_PORT"] = f"{10639 + offset}"
    init_method = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    if "OMPI_COMM_WORLD_RANK" in os.environ:
        rpc.init_rpc(
            f"Test{torch.distributed.get_rank()}",
            rank=torch.distributed.get_rank(),
            world_size=torch.distributed.get_world_size(),
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(init_method=init_method),
        )


@torch_spawn([2])
def basic_rpc():
    init_rpc()
    if torch.distributed.get_rank() != 0:
        rpc.shutdown()
        torch.distributed.barrier()
        return

    model = [nn.Linear(10, 10), nn.ReLU()]
    pipe = PipeRPCWrapper(model, [1, 1], input_device=torch.cuda.current_device(), worker_map=get_worker_map())

    pipe.foreach_worker(register_optimizer, include_self=True)

    inputs = torch.rand(10).cuda()
    output = pipe(inputs)
    loss = output.mean()
    loss.backward()

    pipe.foreach_worker(step_optimizer, include_self=True)

    pipe.eval()

    rpc.shutdown()
    torch.distributed.barrier()


def register_optimizer(ctx, model):
    if len(list(model.parameters())) > 0:
        model.optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    else:
        model.optimizer = None


def step_optimizer(ctx, model):
    if model.optimizer:
        model.optimizer.step()


def check_pipe_against_reference(balance, model_constructor, checkpoint="except_last", custom_inputs=None):
    model = model_constructor()
    reference_model = model_constructor()
    for src, dst in zip(model, reference_model):
        dst.load_state_dict(copy.deepcopy(src.state_dict()))

    reference_model = nn.Sequential(*reference_model).cuda()
    nbatch = 100

    pipe_loss = True
    if pipe_loss:
        loss_func = nn.MSELoss()
    else:
        loss_func = None

    pipe = PipeRPCWrapper(
        model,
        balance,
        input_device=torch.cuda.current_device(),
        worker_map=get_worker_map(),
        checkpoint=checkpoint,
        chunks=nbatch,
        loss_func=loss_func,
    )

    pipe.foreach_worker(register_optimizer, include_self=True)
    register_optimizer(None, reference_model)

    inputs = torch.rand(nbatch, 10).cuda()
    target = torch.rand(nbatch, 10).cuda()
    cloned = inputs.clone()
    if pipe_loss:
        output = pipe(inputs, target=target)
    else:
        output = pipe(inputs)
    ref_out = reference_model(inputs)

    left = ref_out.cpu()
    right = output.cpu()
    if not pipe_loss:
        assert torch.allclose(ref_out.cpu(), output.cpu(), atol=1.0e-3)
        assert torch.allclose(ref_out.cpu(), output.cpu(), atol=1.0e-4)
        assert torch.allclose(ref_out.cpu(), output.cpu(), atol=1.0e-5)
        assert torch.allclose(ref_out.cpu(), output.cpu(), atol=1.0e-6)
        assert torch.allclose(ref_out.cpu(), output.cpu(), atol=1.0e-7)
    # assert torch.allclose(ref_out.cpu(), output.cpu(), atol=1.0e-9)
    # assert torch.allclose(ref_out.cpu(), output.cpu(), atol=1.0e-11)
    # if not torch.equal(ref_out.cpu(), output.cpu()):
    # print(f"wat {left.tolist()}, {right.tolist()}, {left.tolist() == right.tolist()}")

    outputs = {"reference": ref_out, "pipe": output}

    ref_loss = nn.MSELoss()(ref_out, target)
    ref_loss.backward()

    if pipe_loss:
        loss = output
    else:
        loss = nn.MSELoss()(output, target)
    loss.backward()

    assert torch.allclose(ref_loss.cpu(), loss.cpu(), atol=1.0e-7)

    pipe.foreach_worker(step_optimizer, include_self=True)
    step_optimizer(None, reference_model.cuda())

    pipe.eval()
    reference_model.eval()

    final_output = pipe(inputs)
    final_ref = reference_model(inputs.cuda())

    # assert torch.equal(final_output.cpu(), final_ref.cpu())
    assert torch.allclose(final_output.cpu(), final_ref.cpu(), atol=1.0e-7)


@torch_spawn([3])
@pytest.mark.parametrize("checkpoint", ["never", "always", "except_last"])
def rpc_optimizer(checkpoint):
    init_rpc({"never": 0, "always": 1, "except_last": 2}[checkpoint])
    if torch.distributed.get_rank() != 0:
        rpc.shutdown()
        torch.distributed.barrier()
        return

    def model_with_reuse():
        reused_1 = nn.Linear(10, 10)
        return [reused_1, nn.ReLU(), reused_1, nn.ReLU(), reused_1, nn.ReLU()]

    check_pipe_against_reference(
        [2, 2, 2],
        lambda: [nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 10), nn.ReLU()],
        checkpoint=checkpoint,
    )
    check_pipe_against_reference([2, 1, 1], model_with_reuse, checkpoint=checkpoint)

    rpc.shutdown()
    torch.distributed.barrier()


@torch_spawn([6])
@pytest.mark.skipif("OMPI_COMM_WORLD_RANK" not in os.environ, reason="mpi required")
def rpc_megatron_reuse():

    from fairscale.nn.model_parallel import layers
    from fairscale.nn.model_parallel.initialize import destroy_model_parallel, initialize_model_parallel

    def make_model_simple():
        return [
            layers.ColumnParallelLinear(10, 10),
            nn.ReLU(),
            layers.RowParallelLinear(10, 10),
            nn.ReLU(),
            layers.ColumnParallelLinear(10, 10),
            nn.ReLU(),
            layers.RowParallelLinear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
        ]

    def make_model_with_reuse():
        column = layers.ColumnParallelLinear(10, 10)
        row = layers.RowParallelLinear(10, 10)
        return [
            column,
            nn.ReLU(),
            row,
            nn.ReLU(),
            column,
            nn.ReLU(),
            row,
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
        ]

    destroy_model_parallel()
    torch.distributed.destroy_process_group()
    torch.distributed.init_process_group("gloo", rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
    initialize_model_parallel(2, 3, model_parallel_backend="nccl", pipeline_backend="mpi")

    init_rpc()
    if get_pipeline_parallel_group().rank() != 0:
        rpc.shutdown()
        torch.distributed.barrier()
        return

    check_pipe_against_reference([4, 4, 2], make_model_simple, "always")
    check_pipe_against_reference([4, 2, 2], make_model_with_reuse)

    rpc.shutdown()
    torch.distributed.barrier()


@torch_spawn([3])
def rpc_reuse_in_final_stage():

    # 'reused' and 'reused2' are located on stage 2, so the backward pass for
    # the final stage will need to first send gradients to stage 2, then receive
    # gradients from stage 2. This tests custom logic to handle reuse of layers
    # in the final stage of the pipeline.

    reused = nn.Linear(10, 10)
    reused2 = nn.Linear(10, 10)
    model = [
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        reused2,
        nn.ReLU(),
        reused,
        nn.ReLU(),
        reused,
        reused2,
        nn.ReLU(),
        reused,
        nn.ReLU(),
    ]
    balance = [2, 3, 4]

    init_rpc()

    if torch.distributed.get_rank() != 0:
        rpc.shutdown()
        torch.distributed.barrier()
        return

    pipe = PipeRPCWrapper(model, balance, worker_map=get_worker_map())

    inputs = torch.rand(10).cuda()
    target = torch.rand(10).cuda()
    output = pipe(inputs)
    nn.MSELoss()(output, target).backward()
    output = pipe(inputs)
    nn.MSELoss()(output, target).backward()
    rpc.shutdown()
    torch.distributed.barrier()


@torch_spawn([3])
def rpc_multiple_tensors():
    class FuseTwo(nn.Module):
        def forward(self, left, right):
            return left + right

    class SplitTwo(nn.Module):
        def forward(self, inputs):
            return (inputs, 2 * inputs)


@torch_spawn([2])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
def construct_only_rank_zero():
    model = [nn.Linear(10, 10), nn.ReLU()]
    if torch.distributed.get_rank() == 0:
        PipeRPCWrapper(model, [1, 1], worker_map=get_worker_map())
        rpc.shutdown()
    else:
        # Must enter rpc loop to complte PipeRPCWrapper constructor above
        rpc.shutdown()

        with pytest.raises(AssertionError):
            PipeRPCWrapper(model, [1, 1], worker_map=get_worker_map())
