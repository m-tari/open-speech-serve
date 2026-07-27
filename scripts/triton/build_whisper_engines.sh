#!/usr/bin/env bash
set -euo pipefail

# Run inside docker/Dockerfile.triton on the target GPU architecture.
TRTLLM_ROOT="${TRTLLM_ROOT:-/opt/TensorRT-LLM}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/workspace/artifacts/triton}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-32}"
MODEL_NAME="${MODEL_NAME:-large-v3-turbo}"
PRECISION="${PRECISION:-float16}"

EXAMPLE_DIR="${TRTLLM_ROOT}/examples/models/core/whisper"
ASSET_DIR="${ARTIFACT_DIR}/assets"
CHECKPOINT_DIR="${ARTIFACT_DIR}/checkpoint"
ENGINE_DIR="${ARTIFACT_DIR}/engines"
MODEL_REPO="${ARTIFACT_DIR}/model_repo"

mkdir -p "${ASSET_DIR}" "${CHECKPOINT_DIR}" "${ENGINE_DIR}"

wget -nc -P "${ASSET_DIR}" \
  https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken
wget -nc -P "${ASSET_DIR}" \
  https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz
wget -nc -P "${ASSET_DIR}" \
  https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt

python3 "${EXAMPLE_DIR}/convert_checkpoint.py" \
  --model_dir "${ASSET_DIR}" \
  --model_name "${MODEL_NAME}" \
  --dtype "${PRECISION}" \
  --output_dir "${CHECKPOINT_DIR}"

trtllm-build \
  --checkpoint_dir "${CHECKPOINT_DIR}/encoder" \
  --output_dir "${ENGINE_DIR}/encoder" \
  --max_batch_size "${MAX_BATCH_SIZE}" \
  --gemm_plugin disable \
  --bert_attention_plugin "${PRECISION}" \
  --max_input_len 3000 \
  --max_seq_len 3000

trtllm-build \
  --checkpoint_dir "${CHECKPOINT_DIR}/decoder" \
  --output_dir "${ENGINE_DIR}/decoder" \
  --max_beam_width 1 \
  --max_batch_size "${MAX_BATCH_SIZE}" \
  --max_seq_len 114 \
  --max_input_len 14 \
  --max_encoder_input_len 3000 \
  --gemm_plugin "${PRECISION}" \
  --bert_attention_plugin "${PRECISION}" \
  --gpt_attention_plugin "${PRECISION}"

rm -rf "${MODEL_REPO}"
cp -r "${TRTLLM_ROOT}/triton_backend/all_models/whisper" "${MODEL_REPO}"
cp -r \
  "${TRTLLM_ROOT}/triton_backend/all_models/inflight_batcher_llm/tensorrt_llm" \
  "${MODEL_REPO}/tensorrt_llm"
cp "${ASSET_DIR}/multilingual.tiktoken" "${MODEL_REPO}/whisper_bls/1/"
cp "${ASSET_DIR}/mel_filters.npz" "${MODEL_REPO}/whisper_bls/1/"

FILL="${TRTLLM_ROOT}/triton_backend/tools/fill_template.py"
python3 "${FILL}" -i "${MODEL_REPO}/tensorrt_llm/config.pbtxt" \
  "triton_backend:tensorrtllm,engine_dir:${ENGINE_DIR}/decoder,encoder_engine_dir:${ENGINE_DIR}/encoder,decoupled_mode:false,max_tokens_in_paged_kv_cache:24000,batch_scheduler_policy:guaranteed_no_evict,batching_strategy:inflight_fused_batching,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:true,triton_max_batch_size:${MAX_BATCH_SIZE},max_queue_delay_microseconds:0,max_beam_width:1,enable_kv_cache_reuse:false,normalize_log_probs:true,enable_chunked_context:false,gpu_device_ids:0,decoding_mode:top_k_top_p,max_queue_size:0,enable_context_fmha_fp32_acc:false,cross_kv_cache_fraction:0.5,encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32,prompt_embedding_table_data_type:TYPE_FP16"
python3 "${FILL}" -i "${MODEL_REPO}/whisper_bls/config.pbtxt" \
  "engine_dir:${ENGINE_DIR}/encoder,n_mels:128,zero_pad:false,triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:false"

echo "TensorRT-LLM Whisper repository ready at ${MODEL_REPO}"
