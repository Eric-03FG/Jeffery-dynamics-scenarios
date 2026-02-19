#!/usr/bin/env julia
# jeffery_cli.jl
# Run from terminal: julia JefferyV1_cli.jl --interactive
# or: julia JefferyV1_cli.jl --gamma 5 --r 7.14 --theta0 1.1 --phi0 0.3 --tmax 80 --saveat 0.01 --outdir results --open

using DifferentialEquations
using StaticArrays
using LinearAlgebra
using Plots
using LaTeXStrings
using Printf
using Dates

# ----------------------------- Plot defaults ------------------------------
default(
    linewidth = 2,
    framestyle = :box,
    grid = true,
    legend = :topright,
    guidefontsize = 12,
    tickfontsize = 10,
    legendfontsize = 10,
    titlefontsize = 12
)

# ----------------------------- Helper functions ---------------------------
bretherton_parameter(r::Real) = (r^2 - 1) / (r^2 + 1)
jeffery_period(gamma::Real, r::Real) = (2π/gamma) * (r + 1/r)

p_from_angles(θ::Real, φ::Real) = @SVector [
    sin(θ) * cos(φ),
    sin(θ) * sin(φ),
    cos(θ)
]

function angles_from_p(p::SVector{3,<:Real})
    px, py, pz = p
    θ = acos(clamp(pz, -1.0, 1.0))
    φ = atan(py, px)
    return θ, φ
end

unit_norm(v) = sqrt(v[1]^2 + v[2]^2 + v[3]^2)

function P_from_theta_phi(θ, φ)
    N = length(θ)
    P = zeros(N, 3)
    @inbounds for i in 1:N
        P[i,1] = sin(θ[i]) * cos(φ[i])
        P[i,2] = sin(θ[i]) * sin(φ[i])
        P[i,3] = cos(θ[i])
    end
    return P
end

function unwrap_angle_deg(φdeg)
    φ_unwrapped = copy(φdeg)
    for i in 2:length(φdeg)
        Δ = φ_unwrapped[i] - φ_unwrapped[i-1]
        if Δ > 180
            φ_unwrapped[i:end] .-= 360
        elseif Δ < -180
            φ_unwrapped[i:end] .+= 360
        end
    end
    return φ_unwrapped
end

# ------------------ Case A: Jeffery equation (vector form) -----------------
function jeffery_xy!(dp, p, params, t)
    γ, β = params

    E = @SMatrix [0.0  γ/2  0.0;
                  γ/2  0.0  0.0;
                  0.0  0.0  0.0]

    Ω = @SVector [0.0, 0.0, -γ/2]

    pvec = @SVector [p[1], p[2], p[3]]
    Ep   = E * pvec
    pEp  = dot(pvec, Ep)
    rhs  = cross(Ω, pvec) + β * (Ep - pEp * pvec)

    dp[1] = rhs[1]; dp[2] = rhs[2]; dp[3] = rhs[3]
end

function renormalize_p!(integrator)
    p = integrator.u
    n = unit_norm(p)
    integrator.u .= p ./ n
end

# ------------------ Case B: Jeffery equation in angles ---------------------
function jeffery_xz_angles!(du, u, params, t)
    γ, β = params
    θ, φ = u
    θc = clamp(θ, 1e-10, π - 1e-10)

    du[1] = γ * (0.5*(1 - β) + β*cos(θc)^2) * cos(φ)
    du[2] = -0.5*γ*(1 + β) * (cos(θc)/sin(θc)) * sin(φ)
end

# ----------------------------- Simulation wrappers -------------------------
function run_case_xy(; γ=1.0, r=5.0, θ0=1.1, φ0=0.3, tspan=(0.0, 40.0), saveat=0.01)
    β = bretherton_parameter(r)
    params = (γ, β)

    p0 = collect(p_from_angles(θ0, φ0))
    prob = ODEProblem(jeffery_xy!, p0, tspan, params)

    cb = PeriodicCallback(renormalize_p!, 0.5; save_positions=(false,false))
    sol = solve(prob, Tsit5(); callback=cb, abstol=1e-10, reltol=1e-10, saveat=saveat)

    P = hcat(sol.u...)'   # N×3
    norms = [unit_norm(u) for u in sol.u]

    θ = similar(sol.t); φ = similar(sol.t)
    for i in eachindex(sol.u)
        th, ph = angles_from_p(@SVector [P[i,1], P[i,2], P[i,3]])
        θ[i] = th; φ[i] = ph
    end

    return sol, (β=β, P=P, θ=θ, φ=φ, norms=norms)
end

function run_case_xz(; γ=1.0, r=5.0, θ0=1.1, φ0=0.3, tspan=(0.0, 40.0), saveat=0.01)
    β = bretherton_parameter(r)
    params = (γ, β)

    u0 = [θ0, φ0]
    prob = ODEProblem(jeffery_xz_angles!, u0, tspan, params)
    sol = solve(prob, Tsit5(); abstol=1e-10, reltol=1e-10, saveat=saveat)

    θ = getindex.(sol.u, 1)
    φ = getindex.(sol.u, 2)
    P = P_from_theta_phi(θ, φ)

    return sol, (β=β, θ=θ, φ=φ, P=P)
end

# ----------------------------- Plot functions ------------------------------
function plot_case_xy(sol, diag)
    γ = sol.prob.p[1]
    β = diag.β
    t = sol.t
    P = diag.P

    main_title = L"u = \dot{\gamma}\, y\, \hat{\mathbf{x}} \quad (\dot{\gamma} = %$γ,\; \beta = %$(round(β,digits=4)))"

    p1 = plot(t, P[:,1], label=L"p_x", xlabel=L"t", ylabel=L"\mathbf{p}(t)")
    plot!(p1, t, P[:,2], label=L"p_y")
    plot!(p1, t, P[:,3], label=L"p_z")
    title!(p1, main_title)

    θdeg = rad2deg.(diag.θ)
    φdeg = unwrap_angle_deg(rad2deg.(diag.φ))

    p2 = plot(t, θdeg, label=L"\theta(t)", xlabel=L"t", ylabel=L"\theta \; [^\circ]")
    p3 = plot(t, φdeg, label=L"\phi(t)", xlabel=L"t", ylabel=L"\phi \; [^\circ]")
    p4 = plot(t, diag.norms, label=L"\|\mathbf{p}\|", xlabel=L"t", ylabel=L"\|\mathbf{p}\|")

    return plot(p1, p2, p3, p4, layout=(2,2), size=(950, 650))
end

function plot_case_xz(sol, diag)
    γ = sol.prob.p[1]
    β = diag.β
    t = sol.t

    main_title = L"u = \dot{\gamma}\, z\, \hat{\mathbf{x}} \quad (\dot{\gamma} = %$γ,\; \beta = %$(round(β,digits=4)))"

    θdeg = rad2deg.(diag.θ)
    φdeg = unwrap_angle_deg(rad2deg.(diag.φ))

    p1 = plot(t, θdeg, label=L"\theta(t)", xlabel=L"t", ylabel=L"\theta \; [^\circ]")
    title!(p1, main_title)

    p2 = plot(t, φdeg, label=L"\phi(t)", xlabel=L"t", ylabel=L"\phi \; [^\circ]")

    return plot(p1, p2, layout=(2,1), size=(900, 600))
end

function plot_on_sphere(P; title="Case", nsphere=60, show_start=true, show_end=true, show_axes=true)
    # Switch backend locally to plotlyjs for 3D
    plotlyjs()

    θs = range(0, π, length=nsphere)
    ϕs = range(0, 2π, length=2nsphere)

    X = [sin(θ)*cos(ϕ) for θ in θs, ϕ in ϕs]
    Y = [sin(θ)*sin(ϕ) for θ in θs, ϕ in ϕs]
    Z = [cos(θ)        for θ in θs, ϕ in ϕs]

    plt = surface(
        X, Y, Z;
        alpha = 0.3,
        colorbar = false,
        seriescolor = RGB(0.85,0.85,0.85),
        linewidth = 0.3,
        linecolor = :white,
        label = "",
        title = title,
        xlims = (-1,1), ylims = (-1,1), zlims = (-1,1),
        aspect_ratio = :equal,
        legend = :outerright
    )

    plot!(plt, P[:,1], P[:,2], P[:,3]; lw=4, color=:red, label="p(t)")

    if show_start
        scatter!(plt, [P[1,1]], [P[1,2]], [P[1,3]]; markersize=3, label="t0")
    end
    if show_end
        scatter!(plt, [P[end,1]], [P[end,2]], [P[end,3]]; markersize=3, label="tf")
    end

    if show_axes
        plot!(plt, [-1,1], [0,0], [0,0]; lw=3, color=:black, label="")
        plot!(plt, [0,0], [-1,1], [0,0]; lw=3, color=:black, label="")
        plot!(plt, [0,0], [0,0], [-1,1]; lw=3, color=:black, label="")
    end

    return plt
end

# ----------------------------- CLI parsing ------------------------------
"""
Parse args like:
--gamma 1.0 --r 7.14 --theta0 1.1 --phi0 0.3 --tmax 80 --saveat 0.01 --outdir results --open
or:
--interactive
"""
function get_arg(args, key; default=nothing)
    i = findfirst(==(key), args)
    if isnothing(i)
        return default
    end
    if i == length(args)
        error("Missing value after $key")
    end
    return args[i+1]
end

function has_flag(args, flag)
    return any(==(flag), args)
end

function prompt_float(msg; default::Float64)
    print("$(msg) [default=$(default)]: ")
    s = readline()
    isempty(strip(s)) && return default
    return parse(Float64, s)
end

function prompt_tuple_tspan(; default_t0=0.0, default_tmax=80.0)
    t0 = prompt_float("t0", default=default_t0)
    tmax = prompt_float("tmax", default=default_tmax)
    return (t0, tmax)
end

function main()
    args = copy(ARGS)

    interactive = has_flag(args, "--interactive")
    openfiles   = has_flag(args, "--open")

    # Defaults: your alga 50x7 um
    γ_default   = 1.0
    r_default   = 50/7
    θ0_default  = 1.1
    φ0_default  = 0.3
    tspan_default = (0.0, 80.0)
    saveat_default = 0.01
    outdir_default = "ResultsV1"

    if interactive
        println("=== Jeffery Scenario 1 (Deterministic) CLI ===")
        γ   = prompt_float("Shear rate γ (1/s)", default=γ_default)
        r   = prompt_float("Aspect ratio r=a/b", default=r_default)
        θ0  = prompt_float("Initial θ0 [rad]", default=θ0_default)
        φ0  = prompt_float("Initial φ0 [rad]", default=φ0_default)
        tspan = prompt_tuple_tspan(default_t0=tspan_default[1], default_tmax=tspan_default[2])
        saveat = prompt_float("saveat (output dt)", default=saveat_default)
        outdir = outdir_default
    else
        γ   = parse(Float64, get_arg(args, "--gamma";  default=string(γ_default)))
        r   = parse(Float64, get_arg(args, "--r";      default=string(r_default)))
        θ0  = parse(Float64, get_arg(args, "--theta0"; default=string(θ0_default)))
        φ0  = parse(Float64, get_arg(args, "--phi0";   default=string(φ0_default)))
        t0  = parse(Float64, get_arg(args, "--t0";     default=string(tspan_default[1])))
        tmax= parse(Float64, get_arg(args, "--tmax";   default=string(tspan_default[2])))
        saveat = parse(Float64, get_arg(args, "--saveat"; default=string(saveat_default)))
        outdir = get_arg(args, "--outdir"; default=outdir_default)
        tspan = (t0, tmax)
    end

    β = bretherton_parameter(r)
    T = jeffery_period(γ, r)

    println("\nParameters:")
    @printf("  γ      = %.6g 1/s\n", γ)
    @printf("  r      = %.6g\n", r)
    @printf("  β      = %.6g\n", β)
    @printf("  θ0     = %.6g rad\n", θ0)
    @printf("  φ0     = %.6g rad\n", φ0)
    @printf("  tspan  = (%.6g, %.6g)\n", tspan[1], tspan[2])
    @printf("  saveat = %.6g\n", saveat)
    @printf("  T_Jeffery (theoretical) = %.6g s\n\n", T)

    # Prepare output folder
    mkpath(outdir)
    stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    prefix = joinpath(outdir, "jeffery_$(stamp)_γ$(γ)_r$(round(r,digits=3))")

    # Use GR backend for 2D
    gr()

    # --- Run & save Case A ---
    sol_xy, diag_xy = run_case_xy(; γ=γ, r=r, θ0=θ0, φ0=φ0, tspan=tspan, saveat=saveat)
    fig_xy = plot_case_xy(sol_xy, diag_xy)
    file_xy = prefix * "_caseA_xy.png"
    savefig(fig_xy, file_xy)
    println("Saved: $file_xy")

    # --- Run & save Case B ---
    sol_xz, diag_xz = run_case_xz(; γ=γ, r=r, θ0=θ0, φ0=φ0, tspan=tspan, saveat=saveat)
    fig_xz = plot_case_xz(sol_xz, diag_xz)
    file_xz = prefix * "_caseB_xz.png"
    savefig(fig_xz, file_xz)
    println("Saved: $file_xz")

    # --- 3D spheres (Plotly) -> save as HTML ---
    sphere_xy = plot_on_sphere(diag_xy.P; title="Case A")
    sphere_xz = plot_on_sphere(diag_xz.P; title="Case B")
    fig_spheres = plot(sphere_xy, sphere_xz, layout=(1,2), size=(900,500))

    file_html = prefix * "_spheres.html"
    savefig(fig_spheres, file_html)
    println("Saved: $file_html")

    # Back to GR
    gr()

    if openfiles
        println("\nOpening outputs with xdg-open...")
        run(`xdg-open $file_xy`)
        run(`xdg-open $file_xz`)
        run(`xdg-open $file_html`)
    end

    println("\nDone.")
end

main()
