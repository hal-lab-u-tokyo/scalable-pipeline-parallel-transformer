import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import NamedTuple

def eval_error(
    saved_actv: torch.Tensor,
    rec_actv: torch.Tensor,
    debug_context: dict,
):
    assert (
        saved_actv.shape == rec_actv.shape
    ), "Shape mismatch between saved and reconstructed activations."

    if debug_context["block_idx"] == 0: 
        return
    
    batch_size = saved_actv.shape[0]
    flat_sav = saved_actv.reshape(batch_size, -1)
    flat_rec = rec_actv.reshape(batch_size, -1)

    cos = F.cosine_similarity(flat_sav, flat_rec, dim=1, eps=1e-15)
    errs = cos.detach().cpu().tolist()

    layer = debug_context["block_idx"]
    debug_context["error_dict"][layer].extend(errs)

                
class Coupling(nn.Module):
    def forward(self, x):
        return x, torch.zeros_like(x)

class Decoupling(nn.Module):
    def forward(self, inputs):
        x, _ = inputs
        return x

# backward_pass_recoverの戻り値を格納するNamedTuple
class RecoverContext(NamedTuple):
    X_2: torch.Tensor
    X_1: torch.Tensor
    Y_1_req_grad: torch.Tensor
    X_2_req_grad: torch.Tensor
    g_Y_1: torch.Tensor
    f_X_2: torch.Tensor

# reversibleblockをラップしてpareprop用のメソッドを提供
class ParePropReversibleBlockWrapper(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        assert hasattr(block, "forward") and callable(block.forward)
        assert hasattr(block, "F") and isinstance(block.F, nn.Module)
        assert hasattr(block, "G") and isinstance(block.G, nn.Module)
        
        self.block = block
        self.F = block.F
        self.G = block.G
        
    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """(X_2, X_1) -> (Y_2, Y_1)"""
        return self.block.forward(inputs)

    def backward_pass_recover(self, Y_2: torch.Tensor, Y_1: torch.Tensor) -> RecoverContext:
        """
        Parepropの活性化復元フェーズ。
        Y から X を復元し、勾配計算に必要な中間グラフを構築。
        """
        
        # G(Y_1)の計算グラフはGのパラメータを計算するのに必要
        with torch.enable_grad():
            Y_1_req_grad = Y_1.detach().requires_grad_()
            g_Y_1 = self.G(Y_1_req_grad)
                       
        # 活性化復元には計算グラフ不要             
        with torch.no_grad():
            X_2 = Y_2 - g_Y_1
        
        # F(X_2)のグラフを構築
        with torch.enable_grad():
            X_2_req_grad = X_2.detach().requires_grad_()
            f_X_2 = self.F(X_2_req_grad)
            
        with torch.no_grad():
            X_1 = Y_1 - f_X_2
            
        return RecoverContext(X_2, X_1, Y_1_req_grad, X_2_req_grad, g_Y_1, f_X_2)
    
    
    def backward_pass_grads(self, ctx: RecoverContext, dY_2: torch.Tensor, dY_1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Parepropの勾配計算フェーズ"""
        X_2, X_1, Y_1_req_grad, X_2_req_grad, g_Y_1, f_X_2 = ctx
        
        #　Gのパラメータ勾配計算
        with torch.enable_grad():
            g_Y_1.backward(dY_2)
            
        with torch.no_grad():
            dY_1_total = dY_1 + Y_1_req_grad.grad
            dX_2_from_G = dY_2
            
        with torch.enable_grad():
            f_X_2.backward(dY_1_total)
            
        with torch.no_grad():
            dX_1 = dY_1_total
            dX_2_from_F = X_2_req_grad.grad
            dX_2 = dX_2_from_G + dX_2_from_F
            
        return dX_2, dX_1

# Parepropを実行するnn.Sequentialの代替モジュール
class ParePropReversibleSequential(nn.Module):
    def __init__(self, modules: list, *, s1: torch.cuda.Stream, s2: torch.cuda.Stream, autocast_dtype: torch.dtype = torch.float32, debug_context: dict | None = None,):
        super().__init__()
        self.layers = nn.ModuleList(modules)
        self.s1 = s1
        self.s2 = s2
        self.autocast_dtype = autocast_dtype
        self.debug_context = debug_context if debug_context is not None else {}
        
    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        actv2, actv1 = inputs
        
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=True):
            current_inputs = (actv2, actv1)
            for layer in self.layers:
                current_inputs = layer.forward(current_inputs)
                       
        return current_inputs

    def pareprop_backward(
        self, 
        Y_N_2: torch.Tensor, 
        Y_N_1: torch.Tensor, 
        dY_N_2: torch.Tensor, 
        dY_N_1: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """
        削除した autograd.Function.backward() のロジック。
        dX (入力勾配) と X (復元入力) の両方を返す。
        """
        layers = self.layers
        s1, s2 = self.s1, self.s2
        autocast_dtype = self.autocast_dtype
        dY = (dY_N_2, dY_N_1)

        # --- ここから ParePropReversibleSequentialFunction.backward と同じ ---
        # events[...].synchronize()だとCPUがGPUのタスクを完了するまで停止する,このCPUとGPUの同期が速度を落とす
        # .wait_event()にすることでCPUが停止せず、GPUに処理全体を一度に投げて実行させられるので早い
        # 通常のプログラムでは、CPUはGPUにやって欲しい作業をTODOリストとしてストリームに書き込む
        # CPUは書き込んだ作業が終わるのを待たないですぐさま次の計算の準備などの実行に移る
        # event.synchronize()だとGPUがCPUに、「自分の仕事が終わるまで何もせず待て」と命令するのでCPUは何もできない
        # stream.wait_event(event)はCPUからGPUへの命令で、GPUへこの作業が終わるまで待てという命令で、CPUは停止しない
        
        
        events = {}
        for i in range(len(layers)):
            events[f"f{i}"] = torch.cuda.Event()
            events[f"b{i}"] = torch.cuda.Event()
            
        current_stream = torch.cuda.current_stream()
        s1.wait_stream(current_stream)
        s2.wait_stream(current_stream)

        with torch.cuda.stream(s1), torch.amp.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=True):
            last_layer = layers[-1]
            prev_ctx = last_layer.backward_pass_recover(Y_N_2, Y_N_1) # Y -> X
            events["f0"].record(s1)

        reversed_layers = list(reversed(layers))
        for i in range(len(layers) - 1):
            this_layer = reversed_layers[i]
            next_layer = reversed_layers[i+1]

            if i % 2 == 0:
                stream_grad, stream_recover = s1, s2
            else:
                stream_grad, stream_recover = s2, s1

            with torch.cuda.stream(stream_grad), torch.amp.autocast(device_type="cuda", dtype=autocast_dtype, enabled=True):
                if i > 0:
                    # .synchronize() -> .wait_event()
                    stream_grad.wait_event(events[f"b{i-1}"]) 
                # .synchronize() -> .wait_event()
                stream_grad.wait_event(events[f"f{i}"])
                dX = this_layer.backward_pass_grads(prev_ctx, *dY)
                dY = dX
                events[f"b{i}"].record(stream_grad)

            with torch.cuda.stream(stream_recover), torch.amp.autocast(device_type="cuda", dtype=autocast_dtype, enabled=True):
                # .synchronize() -> .wait_event()
                stream_recover.wait_event(events[f"f{i}"])
                prev_ctx = next_layer.backward_pass_recover(prev_ctx.X_2, prev_ctx.X_1)
                events[f"f{i+1}"].record(stream_recover)           

        first_layer = layers[0]
        n_layers = len(layers)
        if (n_layers - 1) % 2 == 0:
            stream_grad = s1
        else:
            stream_grad = s2

        with torch.cuda.stream(stream_grad), torch.amp.autocast(device_type="cuda", dtype=autocast_dtype, enabled=True):
            if n_layers > 1:
                # .synchronize() -> .wait_event()
                stream_grad.wait_event(events[f"b{n_layers-2}"])
            # .synchronize() -> .wait_event()
            stream_grad.wait_event(events[f"f{n_layers-1}"])
            dX = first_layer.backward_pass_grads(prev_ctx, *dY)
            events[f"b{n_layers-1}"].record(stream_grad)

        torch.cuda.current_stream().wait_stream(s1)
        torch.cuda.current_stream().wait_stream(s2)
        # --- ここまで ParePropReversibleSequentialFunction.backward と同じ ---

        # 変更点: 2つのタプルを返す
        grad_input = dX  # (grad_actv2, grad_actv1)
        reconstructed_input = (prev_ctx.X_2, prev_ctx.X_1) # 復元された X

        return grad_input, reconstructed_input
    
    def sequential_backward(
            self, 
            Y_N_2: torch.Tensor, 
            Y_N_1: torch.Tensor, 
            dY_N_2: torch.Tensor, 
            dY_N_1: torch.Tensor
        ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    
        layers = self.layers
        autocast_dtype = self.autocast_dtype
        dY = (dY_N_2, dY_N_1)
        
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=autocast_dtype, enabled=True):
            # 最後の層の入力を復元
            last_layer = layers[-1]
            ctx = last_layer.backward_pass_recover(Y_N_2, Y_N_1)
            
            # 勾配計算と次の入力の復元を、層ごとに逐次実行
            for i in range(len(layers)):
                layer_idx = len(layers) - 1 - i
                this_layer = layers[layer_idx]
                
                # 勾配計算 (autograd有効)
                dX = this_layer.backward_pass_grads(ctx, *dY)
                dY = dX # 次のループのための勾配を準備
                
                # 最初の層でなければ、次の層(N-1)の入力を復元
                if layer_idx > 0:
                    next_layer = layers[layer_idx - 1]
                    ctx = next_layer.backward_pass_recover(ctx.X_2, ctx.X_1)
            
            # 最終的な結果
            grad_input = dY # 最後のdXが最初の層の入力勾配
            reconstructed_input = (ctx.X_2, ctx.X_1) # 最後の復元結果が最初の層の入力
            
            return grad_input, reconstructed_input
        
        

        
        
        
        