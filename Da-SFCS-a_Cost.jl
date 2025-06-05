using Pkg

Pkg.add("JuMP")
Pkg.add("Cbc")
Pkg.add("Random")
Pkg.add("JSON")
Pkg.add("Gurobi")

using Gurobi
using JuMP, Cbc, Random, JSON

function solver(
    S::Vector{Int64},
    K_p::Vector{Int64},
    L::Vector{Tuple{Int64,Int64}},
    F::Vector{Int64},
    t_f::Dict{Int64,Int64},
    c_ij::Dict{Tuple{Int64,Int64},Int64},
    x::Dict{Tuple{Int,Int},Int},
    Delta::Vector{Int64},
    N_s::Dict{Int64,Vector{Int64}},
    E_s::Dict{Int64,Vector{Tuple{Int64,Int64}}},
    b_s::Dict{Int64,Int64},
    u_s::Dict{Int64,Int64},
    m_ns::Dict{Tuple{Int64,Int64},Int64},
    p_ns::Dict{Tuple{Int64,Int64},Int64},
    d_s::Dict{Int64,Int64},
    bw_cost::Dict{Tuple{Int64,Int64},Float64},
    vnf_execution_cost::Dict{Tuple{Int64,Int64},Float64},
    benefit_per_service::Float64,
    cost_penalty::Float64,
)

    model = Model(Gurobi.Optimizer)

    # Is service s admitted or not
    @variable(model, a[s in S], Bin)

    # Traffic of s started processing NF n mapped to VNF f at time slot d or not
    @variable(model, y[s in S, f in F, n in N_s[s], d in Delta], Bin)

    # NF n of NS s os mapped to a VNF instance hosted on physical server k from K_p or not
    @variable(model, q[s in S, k in K_p, n in N_s[s]], Bin)

    # NFs n, (n+1) from N_s of NS s are mapped to VNFs hosted on the same physical server k from K_p or not
    @variable(model, h[s in S, k in K_p, n in N_s[s]], Bin)

    # NF o(e) (e[1]) of NS s started the transmission to its successor NF d(e) (e[2]) at time slot d on virtual link e from E_s or not
    @variable(model, θ[s in S, d in Delta, e in E_s[s]], Bin)

    # The virtual link e from E_s is used for transmission between NFs e[1], e[2] of NS s at time slot d from Delta or not
    @variable(model, θ_hat[s in S, d in Delta, e in E_s[s]], Bin)

    # Virtual link e from E_s of NS s is routed through physical link (i,j) from L or not
    @variable(model, l[s in S, e in E_s[s], (i, j) in L], Bin)

    # Total Bandwith Cost
    @variable(model, TotalBandwidthCost >= 0)

    # Total Execution Cost
    @variable(model, TotalVNFExecutionCost >= 0)

    ### Objective ###
    # Keen to Reinforcement Learning (RL). "Reward engineering"
    @objective(model, Max,
        benefit_per_service * sum(a[s] for s in S) -
        cost_penalty * (TotalBandwidthCost + TotalVNFExecutionCost)
    )

    ### (4) -------------------------------------------------------------------------------------------------------------- V

    @constraint(model, [s in S, n in N_s[s]],
        sum(y[s, f, n, d] for f in F, d in Delta) == a[s]
    )

    ### ------------------------------------------------------------------------------------------------------------------

    ### (5) -------------------------------------------------------------------------------------------------------------- V

    @constraint(model, [s in S, n in N_s[s]],
        sum(y[s, f, n, d] * t_f[f] for f in F, d in Delta) == a[s] * m_ns[s, n]
    )

    ### ------------------------------------------------------------------------------------------------------------------

    ### (6) -------------------------------------------------------------------------------------------------------------- V

    for s in S, e in E_s[s]
        for d′ in Delta, d in Delta
            if d′ < d + p_ns[s, e[1]]
                @constraint(model,
                    θ[s, d′, e] ≤ 1 - sum(y[s, f, e[1], d] for f in F)
                )
            end
        end
    end

    ### ------------------------------------------------------------------------------------------------------------------

    ### (7) -------------------------------------------------------------------------------------------------------------- V

    for s in S, e in E_s[s]
        for d′ in Delta, d in Delta
            if d′ < d + d_s[s]
                @constraint(model,
                    sum(y[s, f, e[2], d′] for f in F) ≤ 1 - θ[s, d, e]
                )
            end
        end
    end

    ### ------------------------------------------------------------------------------------------------------------------

    ### (8) -------------------------------------------------------------------------------------------------------------- V

    for s in S
        for i in 1:length(N_s[s])-1  # Consecutive pairs
            n = N_s[s][i]
            next_n = N_s[s][i+1]
            @constraint(model, [d in Delta, d′ in Delta; d′ < d + p_ns[s, n]],
                sum(y[s, f, next_n, d′] for f in F) ≤ 1 - sum(y[s, f, n, d] for f in F)
            )
        end
    end

    ### ------------------------------------------------------------------------------------------------------------------

    ### (9) -------------------------------------------------------------------------------------------------------------- V

    for s in S, n in N_s[s], f in F
        for d in Delta
            d_end = min(d + p_ns[s, n], maximum(Delta))
            for d′ in d:d_end
                @constraint(model,
                    sum(y[s′, f, n′, d′] for s′ in S if s′ ≠ s for n′ in N_s[s′]) <= 1 - y[s, f, n, d]
                )
            end
        end
    end

    ### ------------------------------------------------------------------------------------------------------------------

    ### (10) -------------------------------------------------------------------------------------------------------------- V

    @constraint(model, [f in F, d in Delta],
        sum(y[s, f, n, d] for s in S for n in N_s[s] if m_ns[s, n] == t_f[f]) ≤ 1
    )

    ### ------------------------------------------------------------------------------------------------------------------

    ### (11) -------------------------------------------------------------------------------------------------------------- V

    @constraint(model, [s in S],
        sum(y[s, f, last(N_s[s]), d] * (d + p_ns[s, last(N_s[s])]) for f in F for d in Delta) ≤ u_s[s]
    )

    ### ------------------------------------------------------------------------------------------------------------------

    ### (12) -------------------------------------------------------------------------------------------------------------- V

    @constraint(model, [s in S, n in N_s[s], k in K_p],
        q[s, k, n] <= sum(y[s, f, n, d] * x[f, k] for f in F, d in Delta)   # Previously ==, <= for a more realistic approach
    )

    ### ------------------------------------------------------------------------------------------------------------------

    ### (12.5) -------------------------------------------------------------------------------------------------------------- V

    @constraint(model, [s in S, n in N_s[s]],
        sum(q[s, k, n] for k in K_p) == a[s]
    )

    ### ------------------------------------------------------------------------------------------------------------------

    ### (13) -------------------------------------------------------------------------------------------------------------- V

    # Linear

    for k in K_p, s in S
        for i in 1:(length(N_s[s])-1)  # Consecutive pairs
            n = N_s[s][i]
            next_n = N_s[s][i+1]
            @constraint(model, h[s, k, n] <= q[s, k, n])
            @constraint(model, h[s, k, n] <= q[s, k, next_n])
            @constraint(model, h[s, k, n] >= q[s, k, n] + q[s, k, next_n] - 1)
        end
    end

    ### ------------------------------------------------------------------------------------------------------------------

    ### (14) -------------------------------------------------------------------------------------------------------------- V

    # Linear

    @variable(model, colocated[s in S, e in E_s[s]], Bin)

    @constraint(model, [s in S, e in E_s[s]],
        colocated[s, e] == sum(h[s, k, e[1]] for k in K_p)
    )

    @constraint(model, [s in S, e in E_s[s]],
        sum(θ[s, d, e] for d in Delta) <= 1 - colocated[s, e]
    )
    @constraint(model, [s in S, e in E_s[s]],
        sum(θ[s, d, e] for d in Delta) <= a[s]
    )
    @constraint(model, [s in S, e in E_s[s]],
        sum(θ[s, d, e] for d in Delta) >= (1 - colocated[s, e]) + a[s] - 1
    )

    ### ------------------------------------------------------------------------------------------------------------------

    ### (15) -------------------------------------------------------------------------------------------------------------- V

    @constraint(model, [s in S, e in E_s[s]],
        sum(θ_hat[s, d, e] for d in Delta) == d_s[s] * sum(θ[s, d, e] for d in Delta)
    )

    ### ------------------------------------------------------------------------------------------------------------------

    ### (16) -------------------------------------------------------------------------------------------------------------- V

    @constraint(model, [s in S, e in E_s[s], d in Delta],
        sum(θ_hat[s, d′, e] for d′ in d:min(d + d_s[s] - 1, maximum(Delta))) >= d_s[s] * θ[s, d, e]
    )

    ### ------------------------------------------------------------------------------------------------------------------

    ### (17) -------------------------------------------------------------------------------------------------------------- V

    # Linear

    @variable(model, link_usage[s in S, e in E_s[s], (i, j) in L, d in Delta], Bin)

    @constraint(model, [s in S, e in E_s[s], (i, j) in L, d in Delta],
        link_usage[s, e, (i, j), d] >= l[s, e, (i, j)] + θ_hat[s, d, e] - 1
    )
    @constraint(model, [s in S, e in E_s[s], (i, j) in L, d in Delta],
        link_usage[s, e, (i, j), d] <= l[s, e, (i, j)]
    )
    @constraint(model, [s in S, e in E_s[s], (i, j) in L, d in Delta],
        link_usage[s, e, (i, j), d] <= θ_hat[s, d, e]
    )

    @constraint(model, [(i, j) in L, d in Delta],
        sum(link_usage[s, e, (i, j), d] * b_s[s] for s in S, e in E_s[s]) <= c_ij[(i, j)]
    )

    ### ------------------------------------------------------------------------------------------------------------------

    ### (18) -------------------------------------------------------------------------------------------------------------- V

    @constraint(model, [s in S, e in E_s[s], i in K_p],
        sum(l[s, e, (i, j)] for (i_l, j) in L if i_l == i) -
        sum(l[s, e, (j, i)] for (j, i_l) in L if i_l == i) ==
        q[s, i, e[1]] - q[s, i, e[2]]
    )

    ### ------------------------------------------------------------------------------------------------------------------

    ### (19) -------------------------------------------------------------------------------------------------------------- V

    @constraint(model, [s in S, e in E_s[s], (i, j) in L],
        l[s, e, (i, j)] ≤ 1 - sum(h[s, k, e[1]] for k in K_p)
    )

    ### ------------------------------------------------------------------------------------------------------------------

    ### (20) -------------------------------------------------------------------------------------------------------------- V

    @constraint(model, [s in S, e in E_s[s], (i, j) in L],
        l[s, e, (i, j)] ≤ a[s]
    )

    ### ------------------------------------------------------------------------------------------------------------------

    ### (21) -------------------------------------------------------------------------------------------------------------- V

    @constraint(model,
        TotalBandwidthCost == sum(
            link_usage[s, e, (i, j), d] * b_s[s] * get(bw_cost, (i, j), 0.0)
            for s in S
            for e in E_s[s]
            for (i, j) in L
            for d in Delta
        )
    )

    ### ------------------------------------------------------------------------------------------------------------------

    ### (22) -------------------------------------------------------------------------------------------------------------- V

    vnf_exec_cost_expr = @expression(model,
        sum(
            begin
                k_host = -1
                for k in K_p
                    if get(x, (f_idx, k), 0) == 1
                        k_host = k
                        break
                    end
                end

                if k_host != -1
                    y[s_idx, f_idx, n_id, d_start_proc] * p_ns[s_idx, n_id] * get(vnf_execution_cost, (f_idx, k_host), 0.0)
                else
                    0.0
                end
            end
            for s_idx in S
            for f_idx in F
            for n_id in N_s[s_idx]
            for d_start_proc in Delta
        )
    )
    @constraint(model, DefineTotalVNFExecutionCost, TotalVNFExecutionCost == vnf_exec_cost_expr)

    ### ------------------------------------------------------------------------------------------------------------------

    optimize!(model)

    ### ------------------------------------------------------------------------------------------------------------------

    ### Data for analysis ------------------------------------------------------------------------------------------------

    if termination_status(model) == MOI.OPTIMAL || termination_status(model) == MOI.FEASIBLE_POINT
        #for s in S
        #    a_value = JuMP.value(a[s])
        #    println(a_value)
        #end
        for s in S, (i, j) in L, e in E_s[s]
            l_value = JuMP.value(l[s, e, (i, j)])
            #println((i, j), " - ", e, " - ", l_value)
            if l_value > 0.5
                println((i, j), " - (", m_ns[s, e[1]], ", ", m_ns[s, e[2]], ") - ", e)
            end
        end
        println("")
        for s in S, f in F, n in N_s[s], d in Delta
            y_value = JuMP.value(y[s, f, n, d])
            #println(s, " - ", f, " - ", n, " - ", d, " - ", y_value)
            if y_value > 0.5
                println(s, " - ", f, " - ", n, " - ", d)
            end
        end
        a_total = 0
        for s in S
            a_value = JuMP.value(a[s])
            a_total += a_value
        end
        println(a_total)
        bw_cost_value = JuMP.value(TotalBandwidthCost)
        println(bw_cost_value)
        vnf_execution_cost_value = JuMP.value(TotalVNFExecutionCost)
        println(vnf_execution_cost_value)
    end

    ### ------------------------------------------------------------------------------------------------------------------

end

function runSolver()
    topoFileName = "topologies/big_topology.json"
    situFileName = "situations/10_services.json"

    topologyData = JSON.parsefile(topoFileName)
    situationData = JSON.parsefile(situFileName)

    # Network Services
    S = convert(Vector{Int}, situationData["S"])
    # Bandwith demanded by NS s from S  # Between 16 and 24
    b_s = Dict{Int,Int}(parse(Int, k) => v for (k, v) in situationData["b_s_bandwidth_demand"])
    # Trafic demand of NS s
    w_s = Dict{Int,Int}(parse(Int, k) => v for (k, v) in situationData["w_s_traffic_demand"])

    # Sequence of NFs required by NS s
    N_s = Dict{Int,Vector{Int}}(
        parse(Int, k) => convert(Vector{Int}, v) for (k, v) in situationData["N_s_nf_sequences"]
    )

    # Virtual links that connect the NFs n from NS s
    E_s = Dict{Int,Vector{Tuple{Int,Int}}}(
        parse(Int, k) => [Tuple(link_arr) for link_arr in v] for (k, v) in situationData["E_s_virtual_links"]
    )

    # Type of NF n from N_s of service S s (relates to t_f)
    m_ns = Dict{Tuple{Int,Int},Int}()
    for item in situationData["m_ns_nf_types"]
        m_ns[(item["service"], item["nf"])] = item["type"]
    end

    # Server nodes
    K_p = convert(Vector{Int}, topologyData["K_p"])
    # Network nodes
    K_n = convert(Vector{Int}, topologyData["K_n"])
    # All nodes
    K = convert(Vector{Int}, topologyData["K"])
    # VNF instances
    F = convert(Vector{Int}, topologyData["F"])
    # Physical links
    L = [Tuple(link_arr) for link_arr in topologyData["L"]]
    # Processing Capacity of VNF f
    p_f = Dict{Int,Int}(parse(Int, k) => v for (k, v) in topologyData["p_f_processing_capacity"])
    # VNF Type Mapping
    t_f = Dict{Int,Int}(parse(Int, k) => v for (k, v) in topologyData["t_f_vnf_type_mapping"])

    # Physical Link Capacities
    c_ij = Dict{Tuple{Int,Int},Int}()
    for item in topologyData["c_ij_link_capacities"]
        c_ij[Tuple(item["link"])] = item["capacity"]
    end

    # Bandwith costs
    bw_cost = Dict{Tuple{Int,Int},Float64}()
    for item in topologyData["c_ij_link_capacities"]
        bw_cost[Tuple(item["link"])] = item["cost"]
    end

    # x(f,k) = 1 if VNF instance f is hosted on physical server k
    x = Dict{Tuple{Int,Int},Int}()
    for item in topologyData["x_vnf_hosting"]
        x[(item["vnf_instance"], item["server"])] = Int(item["is_hosted"])
    end

    # Execution costs
    vnf_execution_cost = Dict{Tuple{Int,Int},Float64}()
    for item in topologyData["x_vnf_hosting"]
        vnf_execution_cost[(item["vnf_instance"], item["server"])] = item["cost"]
    end



    # Processing time of the traffic of NS s from S on the NF
    p_ns = Dict{Tuple{Int64,Int64},Int64}()
    for s in S, f in F, n in N_s[s]
        p_ns[s, n] = div(w_s[s], p_f[f], RoundUp)
    end

    # Transmission time of s on virtual link
    d_s = Dict{Int64,Int64}()
    for s in S
        d_s[s] = div(w_s[s], b_s[s], RoundUp)
    end

    Total_Processing_Delay_s = Dict{Int,Int}()
    for s in S
        Total_Processing_Delay_s[s] = sum(p_ns[s, n] for n in N_s[s])
    end

    Total_Transmission_Delay_s = Dict{Int,Int}()
    for s in S
        Total_Transmission_Delay_s[s] = sum(d_s[s] for e in E_s[s])
    end

    # Service deadlines
    u_s = Dict{Int,Int}()
    for s in S
        u_s[s] = ceil(Int, (Total_Processing_Delay_s[s] + Total_Transmission_Delay_s[s]) * (4 / 3))
    end

    all_values = values(u_s)
    highest_value = maximum(all_values)
    lowest_value = minimum(all_values)
    # Timeslots. Defined by highest deadline + 1
    Delta = collect(1:highest_value+lowest_value)

    println(u_s)
    println(p_ns)
    println(d_s)

    # Values to maximize service admission while aimimng for better costs
    benefit_per_service = 100000.0
    cost_penalty = 1.0

    solver(S, K_p, L, F, t_f, c_ij, x, Delta, N_s, E_s, b_s, u_s, m_ns, p_ns, d_s, bw_cost, vnf_execution_cost, benefit_per_service, cost_penalty)
end

runSolver()