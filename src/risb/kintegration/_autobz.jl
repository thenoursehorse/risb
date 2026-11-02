using AutoBZCore, JSON, LinearAlgebra
BLAS.set_num_threads(1)

function emit(message)
    println(JSON.json(message))
    flush(stdout)
end

function integrate(request)
    dimension = Int(request["dimension"])
    prototype = zeros(ComplexF64, Int(request["size"]))
    function integrand(k, _)
        emit(Dict("kind" => "evaluate", "kpoints" => collect(k)))
        response = JSON.parse(readline(stdin))
        complex.(Float64.(response["real"]), Float64.(response["imag"]))
    end
    bz = load_bz(FBZ(dimension), 2pi * Matrix{Float64}(I, dimension, dimension))
    problem = AutoBZProblem(TrivialRep(), IntegralFunction(integrand, prototype), bz)
    algorithm = if request["method"] == "adaptive"
        IAI(AuxQuadGKJL())
    elseif request["method"] == "automatic-uniform"
        AutoPTR(a=Float64(request["rate"]), nmin=Int(request["points"]),
                nmax=Int(request["max_points"]))
    else
        PTR(npt=Int(request["points"]))
    end
    solution = solve(problem, algorithm; abstol=Float64(request["atol"]),
                     reltol=Float64(request["rtol"]),
                     maxiters=Int(request["max_evaluations"]))
    emit(Dict("kind" => "result", "real" => real.(solution.value),
              "imag" => imag.(solution.value),
              "error" => get(solution.stats, :error, nothing),
              "converged" => solution.retcode == AutoBZCore.Success))
end

emit(Dict("kind" => "ready"))
for line in eachline(stdin)
    try
        integrate(JSON.parse(line))
    catch exception
        emit(Dict("kind" => "error", "message" => sprint(showerror, exception)))
    end
end
