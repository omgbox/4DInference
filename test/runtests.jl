using Test
using Random

include(joinpath(@__DIR__, "..", "src", "FourDInference.jl"))
using .FourDInference

println("Running tests...")

@testset "Data Module" begin
    @testset "Dataset generation" begin
        X, y, diff = generate_dataset(100; seed=42)
        @test size(X, 1) == 100
        @test size(X, 2) == 3
        @test length(y) == 100
        @test length(diff) == 100
        @test all(1 .<= y .<= 6)
        @test all(1 .<= diff .<= 3)
    end

    @testset "DataLoader" begin
        X, y, diff = generate_dataset(100; seed=42)
        loader = DataLoader(X, y, diff; batch_size=32)
        X_batch, y_batch, diff_batch = next_batch!(loader)
        @test size(X_batch, 1) == 32
        @test length(y_batch) == 32
        @test length(diff_batch) == 32
    end
end

@testset "Memory Module" begin
    @testset "SORN Memory creation" begin
        mem = create_sorn_memory!(3; n_neurons=10, seed=42)
        @test mem.n_neurons == 10
        @test length(mem.state) == 10
        @test size(mem.W) == (10, 10)
    end

    @testset "Memory read/write" begin
        mem = create_sorn_memory!(3; n_neurons=10, seed=42)
        input = randn(3)
        old_state = copy(mem.state)
        surprise_write!(mem, input)
        @test mem.state != old_state

        state = read_memory(mem)
        @test length(state) == 10
    end
end

@testset "Phase Module" begin
    @testset "PhaseManager creation" begin
        pm = PhaseManager(; seed=42)
        @test pm.current == RETRIEVE
        @test size(pm.embedding) == (4, 8)
    end

    @testset "Phase encoding/decoding" begin
        pm = PhaseManager(; seed=42)
        embed = encode_phase(pm, REASON)
        @test length(embed) == 8

        logits = [1.0, 0.0, 0.0, 0.0]
        phase, probs = decode_phase(pm, logits)
        @test phase == RETRIEVE
        @test sum(probs) ≈ 1.0
    end

    @testset "Phase update" begin
        pm = PhaseManager(; seed=42)
        update_phase!(pm, REASON)
        @test pm.current == REASON
        @test length(pm.phase_history) == 1
    end
end

@testset "FiLM Module" begin
    @testset "FilmLayer creation" begin
        fl = FilmLayer(8, 16; seed=42)
        @test size(fl.gamma_w) == (16, 8)
        @test size(fl.beta_w) == (16, 8)
    end

    @testset "FiLM forward" begin
        fl = FilmLayer(8, 16; seed=42)
        features = randn(16)
        phase_embed = randn(8)
        result = film_forward(fl, features, phase_embed)
        @test length(result) == 16
    end

    @testset "FiLM initialization" begin
        fl = FilmLayer(8, 16; seed=42)
        film_init!(fl; gamma_init=1.0, beta_init=0.0)
        @test all(fl.gamma_b .≈ 1.0)
        @test all(fl.beta_b .≈ 0.0)
    end
end

@testset "Slices Module" begin
    @testset "Slice creation" begin
        s = Slice(3, 16, 8, 2, "TEST"; seed=42)
        @test s.name == "TEST"
        @test length(s.weights) == 2
        @test length(s.biases) == 2
    end

    @testset "Slice forward" begin
        s = Slice(3, 16, 8, 2, "TEST"; seed=42)
        input = randn(3)
        output = slice_forward(s, input)
        @test length(output) == 8
    end

    @testset "Create all slices" begin
        slices = create_slices!(3, 16, 8; seed=42)
        @test length(slices) == 4
        @test slices[1].name == "RETRIEVE"
        @test slices[2].name == "REASON"
        @test slices[3].name == "PLAN"
        @test slices[4].name == "COMPRESS"
    end
end

@testset "Router Module" begin
    @testset "Router creation" begin
        router = create_router!(3 + 10 + 8, 16; seed=42)
        @test router.hidden_dim == 16
    end

    @testset "Router forward" begin
        router = create_router!(21, 16; seed=42)
        input = randn(21)
        output = router_forward(router, input)
        @test output.chosen_slice in 1:3
        @test 0.0 <= output.confidence <= 1.0
        @test length(output.slice_probs) == 3
        @test length(output.phase_probs) == 4
        @test sum(output.slice_probs) ≈ 1.0
        @test sum(output.phase_probs) ≈ 1.0
    end
end

@testset "Inference Module" begin
    @testset "Model creation" begin
        model = create_model!(3; hidden_dim=8, memory_neurons=5, max_steps=3, seed=42)
        @test model.input_dim == 3
        @test model.memory_summary_dim == 5
        @test model.max_steps == 3
        @test length(model.slice_proj_ws) == 4
        @test length(model.slice_proj_bs) == 4
        @test size(model.slice_proj_ws[1]) == (8, 3)
    end

    @testset "Per-slice projections are different" begin
        model = create_model!(3; hidden_dim=8, memory_neurons=5, max_steps=3, seed=42)
        for i in 1:4
            for j in (i+1):4
                @test model.slice_proj_ws[i] != model.slice_proj_ws[j]
            end
        end
    end

    @testset "Forward pass" begin
        model = create_model!(3; hidden_dim=8, memory_neurons=5, max_steps=3, seed=42)
        input = randn(3)
        trace = forward(model, input)
        @test trace.total_steps >= 1
        @test trace.total_steps <= 3
        @test length(trace.final_output) == 6
        @test length(trace.phase_history) == trace.total_steps
        @test length(trace.slice_history) == trace.total_steps
    end

    @testset "Batch forward" begin
        model = create_model!(3; hidden_dim=8, memory_neurons=5, max_steps=3, seed=42)
        X = randn(10, 3)
        outputs, traces = forward(model, X)
        @test size(outputs) == (10, 6)
        @test length(traces) == 10
    end
end

@testset "Training Module" begin
    @testset "Loss computation" begin
        model = create_model!(3; hidden_dim=8, memory_neurons=5, max_steps=3, seed=42)
        X = randn(10, 3)
        y = rand(1:6, 10)
        diff = rand(1:3, 10)

        total_loss, L_task, L_diff, L_entropy, L_phase, traces = compute_loss(
            model, X, y, diff
        )

        @test total_loss >= 0.0
        @test L_task >= 0.0
        @test L_diff >= 0.0
        @test length(traces) == 10
    end

    @testset "Evaluation" begin
        model = create_model!(3; hidden_dim=8, memory_neurons=5, max_steps=3, seed=42)
        X = randn(20, 3)
        y = rand(1:6, 20)
        diff = rand(1:3, 20)

        metrics = evaluate(model, X, y, diff)
        @test 0.0 <= metrics["accuracy"] <= 1.0
        @test 0.0 <= metrics["easy_accuracy"] <= 1.0
        @test 0.0 <= metrics["medium_accuracy"] <= 1.0
        @test 0.0 <= metrics["hard_accuracy"] <= 1.0
        @test metrics["easy_avg_steps"] >= 0.0
        @test metrics["medium_avg_steps"] >= 0.0
        @test metrics["hard_avg_steps"] >= 0.0
    end
end

@testset "Full Pipeline" begin
    @testset "Training and evaluation" begin
        X_train, y_train, diff_train = generate_dataset(100; seed=42)
        X_val, y_val, diff_val = generate_dataset(50; seed=123)

        model = create_model!(3; hidden_dim=8, memory_neurons=5, max_steps=3, seed=42)
        train_loader = DataLoader(X_train, y_train, diff_train; batch_size=32)

        history = train!(model, train_loader, X_val, y_val, diff_val;
                         epochs=5, lr=0.001, print_every=2)

        metrics = evaluate(model, X_val, y_val, diff_val)
        @test metrics["accuracy"] >= 0.0
        @test length(history["loss"]) == 5
    end
end

println("\nAll tests passed!")
