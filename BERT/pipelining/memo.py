from abc import ABC, abstractmethod

class _PipelineStageBase(ABC):

    def __init__(self, num_stages, args, kwargs):
        """ 
        パイプラインステージの初期化
        ステージ数だけでマイクロバッチ数は関係ない
        """
        pass
        
    @abstractmethod
    def _prepare_forward_infra(
        self, num_microbatches: int, args, kwargs):
        pass
        
    def _prepare_backward_infra(
        self, num_microbatches: int, args, kwargs):
        """ 
        各マイクロバッチごとに必要なインフラを準備する
        """
        pass

    def forward_one_chunk(
        self, fwd_chunk_id: int, args, kwargs):
        """ 
        マイクロバッチの一部を処理する 
        なんか，ステージの入力と出力保存してね…？
        """
        pass
        # self.fwd_cache[fwd_chunk_id] = (
        #     output_tuple,  # stage_output
        #     flatten_input_tensors,  # input_values
        # )

    def backward_one_chunk(
        self, bwd_chunk_id: int, args, kwargs):
        """ マイクロバッチの一部を逆伝播する 
        保存したステージの入力と出力を使ってね…？
        """

        pass
        # (
        #     stage_output,
        #     input_values,
        # ) = self.fwd_cache.pop(bwd_chunk_id)
        

class PipelineStage(_PipelineStageBase):

    def __init__(self, num_stages, args, kwargs):
        super().__init__(num_stages, args, kwargs)
        pass

    def _prepare_forward_infra(self, num_microbatches: int, args, kwargs):
        pass
        # self.args_recv_info[chunk_id] = recv_infos