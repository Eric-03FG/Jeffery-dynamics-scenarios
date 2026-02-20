# Jeffery Scenario 1 (Deterministic)
# - Case A: Simple shear in xy plane: u = γ y e_x
#   Integrate Jeffery equation in vector form for director p(t) on S².
#
# - Case B: Simple shear in xz plane: u = γ z e_x
#   Integrate Jeffery equation in (θ, φ) with coupled ODEs.
#
# Dependencies:
#   DifferentialEquations.jl, StaticArrays.jl, LinearAlgebra, Plots.jl, LaTeXStrings.jl

using DifferentialEquations
using StaticArrays
using LinearAlgebra
using Plots
using LaTeXStrings

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

## ----------------------------- Helper functions ---------------------------
"""
bretherton_parameter(r)

Bretherton parameter β for an axisymmetric spheroid with aspect ratio r = a/b.
For prolate spheroid (elongated), r > 1 → β ∈ (0,1).
β = (r^2 - 1) / (r^2 + 1)
"""
bretherton_parameter(r::Real) = (r^2 - 1) / (r^2 + 1)

"""
p_from_angles(θ, φ)

Map spherical angles (θ, φ) to a unit director vector p (SVector{3}).

Conventions:
- θ: polar angle from +z, θ ∈ [0, π]
- φ: azimuth angle in x-y plane from +x, φ ∈ (-π, π]
"""
p_from_angles(θ::Real, φ::Real) = @SVector [
    sin(θ) * cos(φ),
    sin(θ) * sin(φ),
    cos(θ)
]

"""
angles_from_p(p)

Map unit director vector p to (θ, φ) with:
- θ ∈ [0, π]
- φ ∈ (-π, π]
"""
function angles_from_p(p::SVector{3,<:Real})
    px, py, pz = p
    θ = acos(clamp(pz, -1.0, 1.0))
    φ = atan(py, px)
    return θ, φ
end

"""
unit_norm(v)

Compute Euclidean norm of a 3-vector (works for Vector or SVector).
"""
unit_norm(v) = sqrt(v[1]^2 + v[2]^2 + v[3]^2)

"""
P_from_theta_phi(θ, φ)

Build an N×3 matrix P where each row is p(t) from θ(t), φ(t).
"""
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
"""
jeffery_xy!(dp, p, params, t)

Jeffery ODE for simple shear u = γ y e_x (shear in the xy plane).

State:
- p = [px, py, pz]  (director, ideally unit length)

Parameters:
- params = (γ, β)
    γ: shear rate
    β: Bretherton parameter

Equation:
    ṗ = Ω × p + β( E p - (pᵀ E p) p )

where:
    ∇u = [0 γ 0; 0 0 0; 0 0 0]
    E = 0.5(∇u + ∇uᵀ) and Ω = 0.5 curl(u) = (0,0,-γ/2)
"""
function jeffery_xy!(dp, p, params, t)
    γ, β = params

    # Rate-of-strain tensor E for u = (γ y, 0, 0)
    E = @SMatrix [0.0  γ/2  0.0;
                  γ/2  0.0  0.0;
                  0.0  0.0  0.0]

    # Background angular velocity (vorticity/2)
    Ω = @SVector [0.0, 0.0, -γ/2]

    # Convert state p to an SVector for fast linear algebra
    pvec = @SVector [p[1], p[2], p[3]]

    # Jeffery RHS
    Ep   = E * pvec
    pEp  = dot(pvec, Ep)
    rhs  = cross(Ω, pvec) + β * (Ep - pEp * pvec)

    dp[1] = rhs[1]
    dp[2] = rhs[2]
    dp[3] = rhs[3]
end

"""
renormalize_p!(integrator)

Project p back to unit sphere to avoid small numerical drift in ||p||.
This is optional because Jeffery dynamics preserve ||p|| analytically,
but time integration may accumulate floating-point error.
"""
function renormalize_p!(integrator)
    p = integrator.u
    n = unit_norm(p)
    integrator.u .= p ./ n
end

# ------------------ Case B: Jeffery equation in angles ---------------------
"""
jeffery_xz_angles!(du, u, params, t)

Jeffery ODE in spherical angles for shear u = γ z e_x (shear in xz plane),
using the user-provided system:

  dθ/dt = γ [ 1/2(1-β) + β cos²θ ] cosφ
  dφ/dt = -γ/2 (1+β) cotθ sinφ

State:
- u = [θ, φ]

Parameters:
- params = (γ, β)

Note: cotθ blows up at θ = 0, π. We clamp θ away from poles numerically.
"""
function jeffery_xz_angles!(du, u, params, t)
    γ, β = params
    θ, φ = u

    # Avoid division by ~0 near poles
    θc = clamp(θ, 1e-10, π - 1e-10)

    du[1] = γ * (0.5*(1 - β) + β*cos(θc)^2) * cos(φ)
    du[2] = -0.5*γ*(1 + β) * (cos(θc)/sin(θc)) * sin(φ)
end

#%% ----------------------------- Simulation wrappers -------------------------
"""
run_case_xy(; γ, r, θ0, φ0, tspan, saveat)

Run Case A (vector Jeffery in xy shear) and return solution + diagnostics.
"""
function run_case_xy(; γ=1.0, r=5.0, θ0=1.1, φ0=0.3, tspan=(0.0, 40.0), saveat=0.01)
    β = bretherton_parameter(r)
    params = (γ, β)

    p0 = collect(p_from_angles(θ0, φ0))   # ODEProblem expects a mutable Vector
    prob = ODEProblem(jeffery_xy!, p0, tspan, params)

    # Renormalize occasionally (e.g., every 0.5 units of time)
    cb = PeriodicCallback(renormalize_p!, 0.5; save_positions=(false,false))

    sol = solve(prob, Tsit5(); callback=cb, abstol=1e-10, reltol=1e-10, saveat=saveat)

    # Pack diagnostics
    P = hcat(sol.u...)'                  # N×3 matrix
    norms = [unit_norm(u) for u in sol.u]

    # Angles derived from p(t)
    θ = similar(sol.t)
    φ = similar(sol.t)
    for i in eachindex(sol.u)
        th, ph = angles_from_p(@SVector [P[i,1], P[i,2], P[i,3]])
        θ[i] = th
        φ[i] = ph
    end

    return sol, (β=β, P=P, θ=θ, φ=φ, norms=norms)
end

"""
run_case_xz(; γ, r, θ0, φ0, tspan, saveat)

Run Case B (angle system in xz shear) and return solution.
"""
function run_case_xz(; γ=1.0, r=5.0, θ0=1.1, φ0=0.3, tspan=(0.0, 40.0), saveat=0.01)
    β = bretherton_parameter(r)
    params = (γ, β)

    u0 = [θ0, φ0]
    prob = ODEProblem(jeffery_xz_angles!, u0, tspan, params)
    sol = solve(prob, Tsit5(); abstol=1e-10, reltol=1e-10, saveat=saveat)

    θ = getindex.(sol.u, 1)
    φ = getindex.(sol.u, 2)

    # Build P(t) for sphere plotting
    P = P_from_theta_phi(θ, φ)

    return sol, (β=β, θ=θ, φ=φ, P=P)
end
#%% ----------------------------- Plot functions ------------------------------
"""
plot_case_xy(sol, diag; title_prefix="")

Generate a 2×2 layout:
1) p components
2) θ(t)
3) φ(t)
4) ||p||(t) (should be ~1)
"""
function plot_case_xy(sol, diag)
    γ = sol.prob.p[1]
    β = diag.β
    t = sol.t
    P = diag.P

    main_title = L"u = \dot{\gamma}\, y\, \hat{\mathbf{x}} \quad (\dot{\gamma} = %$γ,\; \beta = %$(round(β,digits=4)))"

    # p components
    p1 = plot(t, P[:,1], label=L"p_x", xlabel=L"t", ylabel=L"\mathbf{p}(t)")
    plot!(p1, t, P[:,2], label=L"p_y")
    plot!(p1, t, P[:,3], label=L"p_z")
    title!(p1, main_title)

    # Convert angles to degrees
    θdeg = rad2deg.(diag.θ)
    φdeg = rad2deg.(diag.φ)

    # Wrap φ to [-180, 180]
    φdeg = unwrap_angle_deg(φdeg)

    # θ(t) in degrees
    p2 = plot(t, θdeg,
              label=L"\theta(t)",
              xlabel=L"t",
              ylabel=L"\theta \; [^\circ]")

    # φ(t) in degrees (wrapped)
    p3 = plot(t, φdeg,
              label=L"\phi(t)",
              xlabel=L"t",
              ylabel=L"\phi \; [^\circ]")

    # norm
    p4 = plot(t, diag.norms,
              label=L"\|\mathbf{p}\|",
              xlabel=L"t",
              ylabel=L"\|\mathbf{p}\|")

    return plot(p1, p2, p3, p4, layout=(2,2), size=(950, 650))
end

"""
plot_case_xz(sol, diag; title_prefix="")

Generate a 2×1 layout:
1) θ(t)
2) φ(t)
"""
function plot_case_xz(sol, diag)
    γ = sol.prob.p[1]
    β = diag.β
    t = sol.t

    main_title = L"u = \dot{\gamma}\, z\, \hat{\mathbf{x}} \quad (\dot{\gamma} = %$γ,\; \beta = %$(round(β,digits=4)))"

    θdeg = rad2deg.(diag.θ)
    φdeg = rad2deg.(diag.φ)

    # Wrap φ to [-180, 180]
    φdeg = unwrap_angle_deg(φdeg)

    p1 = plot(t, θdeg,
              label=L"\theta(t)",
              xlabel=L"t",
              ylabel=L"\theta \; [^\circ]")
    title!(p1, main_title)

    p2 = plot(t, φdeg,
              label=L"\phi(t)",
              xlabel=L"t",
              ylabel=L"\phi \; [^\circ]")

    return plot(p1, p2, layout=(2,1), size=(900, 600))
end


"""
plot_on_sphere(P; title=L"", nsphere=50, show_start=true, show_end=true)

Plot the director trajectory P (N×3) on the unit sphere.

Inputs:
- P :: Matrix{<:Real} with size (N,3), rows are [px py pz].
- nsphere :: sphere mesh resolution
"""

function plot_on_sphere(P; title="Case", nsphere=60, show_start=true, show_end=true, show_axes=true)

    plotlyjs()  # only for this 3D plot

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

    # Trajectory
    plot!(plt, P[:,1], P[:,2], P[:,3]; lw=4, color=:red, label="p(t)")

    # Start/end points
    if show_start
        scatter!(plt, [P[1,1]], [P[1,2]], [P[1,3]]; markersize=3, label="t0")
    end
    if show_end
        scatter!(plt, [P[end,1]], [P[end,2]], [P[end,3]]; markersize=3, label="tf")
    end

    # Axes lines
    if show_axes
        plot!(plt, [-1,1], [0,0], [0,0]; lw=3, color=:black, label="")
        plot!(plt, [0,0], [-1,1], [0,0]; lw=3, color=:black, label="")
        plot!(plt, [0,0], [0,0], [-1,1]; lw=3, color=:black, label="")
    end

    return plt
end

#%% ----------------------------- Main execution ------------------------------
gr()
γ = 1.0
r = 50/7          # ≈ 7.142857
θ0 = 1.1
φ0 = 0.3
tspan = (0.0, 80.0)

# Run and plot Case A
sol_xy, diag_xy = run_case_xy(; γ=γ, r=r, θ0=θ0, φ0=φ0, tspan=tspan, saveat=0.01)
fig_xy = plot_case_xy(sol_xy, diag_xy)
display(fig_xy)

# Run and plot Case B
sol_xz, diag_xz = run_case_xz(; γ=γ, r=r, θ0=θ0, φ0=φ0, tspan=tspan, saveat=0.01)
fig_xz = plot_case_xz(sol_xz, diag_xz)
display(fig_xz)

# --- 3D spheres in PlotlyJS (only) ---
sphere_xy = plot_on_sphere(diag_xy.P; title="Case A: u = γ y x̂")
sphere_xz = plot_on_sphere(diag_xz.P; title="Case B: u = γ z x̂")

fig_spheres = plot(sphere_xy, sphere_xz, layout=(1,2), size=(900,500))
display(fig_spheres)

# --- back to GR for anything else ---
gr()
