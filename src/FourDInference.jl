module FourDInference

export PhaseType, RETRIEVE, REASON, PLAN, COMPRESS
export SORNMemory, Sample, DataLoader, InferenceStep, InferenceTrace, InferenceStepTrace
export FourDModel, FourDRouter, RouterOutput, Slice, FilmLayer, PhaseManager
export read_memory, surprise_write!, encode_phase, decode_phase, update_phase!
export slice_forward, slice_forward_cached, film_forward, film_init!, router_forward, softmax_stable
export forward, forward_single, forward_sequence
export create_model!, generate_dataset, generate_sequence_dataset
export train!, train_with_backprop!, train_curriculum!
export evaluate, classify, softmax
export LABEL_NAMES, N_SLICES, N_ROUTED_SLICES, N_PHASES, PHASE_EMBED_DIM, COMPRESS_IDX
export compute_ranking_gradient!, compute_slice_loss_only, forward_all_routed_fast, router_forward_with_cache

include("data.jl")
include("memory.jl")
include("phase.jl")
include("film.jl")
include("slices.jl")
include("router.jl")
include("inference.jl")
include("training.jl")
include("training_backprop.jl")

end
