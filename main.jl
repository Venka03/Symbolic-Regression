# import Pkg; Pkg.add("Latexify"); Pkg.add("Plots"); Pkg.add("StatsBase"); Pkg.add("CSV"); Pkg.add("DataFrames")  # for first time run
using Latexify
using Random
using StatsBase: sample
using CSV
using DataFrames
using Plots
using Statistics

function load_and_clean_csv(file_path::String)
    # CSV.read is incredibly fast and automatically handles headers and missing data
    df = CSV.read(file_path, DataFrame)

    # Instantly convert the clean DataFrame into a Float64 Matrix
    return Matrix{Float64}(df)
end

function prepare_data(file_path::String)
    data = load_and_clean_csv(file_path)
    println("Loaded clean CSV data: ", size(data))

    X = data[:, 1:end-1]
    y = data[:, end]

    return X, y
end

mutable struct Variable
    name::String
    value::Float64
end

Base.show(io::IO, var::Variable) = print(io, var.name)

mutable struct Operation
    name::String
end

Base.show(io::IO, op::Operation) = print(io, op.name)

mutable struct TreeNode
    value::Union{Float64, Operation, Variable}
    children::Vector{TreeNode}
end

mutable struct Tree
    head::TreeNode
    variables::Vector{Variable}
    fitness::Float64
    predictions::Vector{Float64}
end

Base.isless(a::Tree, b::Tree) = a.fitness < b.fitness

mutable struct Population
    individuals::Vector{Tree}
    size::Int
end


function stringNode(node::TreeNode)
    if typeof(node.value) == Variable
        return "(" * node.value.name * ")"
    end
    if typeof(node.value) != Operation
        return "(" * string(node.value) * ")"
    end

    op1 = stringNode(node.children[1])
    if length(node.children) == 1
        return "(" * node.value.name * op1 * ")"
    end

    op2 = stringNode(node.children[2])
    return "(" * op1 * node.value.name * op2 * ")"
end

function stringTree(tree::Tree)
    stringNode(tree.head)
end

function printBeautifulTree(tree::Tree)
    string_tree = stringTree(tree)
    # Convert the string into a mathematical expression
    expressionTree = Meta.parse(string_tree)
    # Render it
    display(latexify(expressionTree))
end

function performOperation(operation::Operation, operand1::Float64, operand2=nothing)
    if operation.name == "sqrt"
        operand1 < 0 ? throw(DomainError(operand1)) : sqrt(operand1)
    elseif operation.name == "log"
        operand1 <= 0 ? throw(DomainError(operand1)) : log(operand1)
    elseif operation.name == "sin"
        sin(operand1)
    elseif operation.name == "cos"
        cos(operand1)
    elseif operation.name == "abs"
        abs(operand1)   
    elseif operation.name == "floor"
        floor(operand1)
    elseif operation.name == "ceil"
        ceil(operand1)
    elseif operand2 !== nothing
        if operation.name == "+"
            operand1 + operand2
        elseif operation.name == "-"
            operand1 - operand2
        elseif operation.name == "*"
            operand1 * operand2
        elseif operation.name == "/"
            operand2 == 0 ? throw(DivideError()) : operand1 / operand2
        end
    else
        # catch error and make fitness be infinity
        throw("$(operation.name) is a binary operation and requires two operands")
    end
end

function computeNode(node::TreeNode)
    if typeof(node.value) == Variable
        return node.value.value
    end
    if typeof(node.value) != Operation
        return node.value
    end

    op1 = computeNode(node.children[1])
    if length(node.children) == 1
        return performOperation(node.value, op1)
    end

    op2 = computeNode(node.children[2])
    return performOperation(node.value, op1, op2)
end

function computeTree(tree::Tree)
    computeNode(tree.head)
end

function generateOperation()
    randNum = rand()
    if randNum < 1/11
        Operation("sqrt")
    elseif randNum < 2/11
        Operation("log")
    elseif randNum < 3/11
        Operation("sin")
    elseif randNum < 4/11
        Operation("cos")
    elseif randNum < 5/11
        Operation("abs")
    elseif randNum < 6/11
        Operation("floor")
    elseif randNum < 7/11
        Operation("ceil")
    elseif randNum < 8/11
        Operation("+")
    elseif randNum < 9/11
        Operation("-")
    elseif randNum < 10/11
        Operation("*")
    else
        Operation("/")
    end
end

function generateNumber(variables::Vector{Variable})
    randNum = rand()
    if randNum < 1/4  # generate float from 0 to 10
        10*rand()
    elseif randNum < 1/2  # generate int from 0 to 10
        float(rand(0:10))
    else
        rand(variables)  # take randomly variable
    end
end

function generateVariables(num::Int)
    alphabet = "xyznkmuvwabcdefghijpqrtslo"
    alphabetVector = Vector{String}()
    for el in alphabet
        push!(alphabetVector, string(el))
    end
    variableNames = alphabetVector[1:num]

    variables = Vector{Variable}()

    for name in variableNames
        push!(variables, Variable(name, 0))  # make 0 value as default
    end
    return variables
end

function generateNode(variables::Vector{Variable}, minDepth::Int,
                                                  depth::Int, full::Bool)
    if depth == 0
        TreeNode(generateNumber(variables), Vector{TreeNode}())
    else
      if rand() < 0.5 || full || minDepth > 0
          operation = generateOperation()
          node = TreeNode(operation, Vector{TreeNode}())
          child1 = generateNode(variables, minDepth-1, depth-1, full)
          node.children = [child1]
          if operation.name in ["+", "-", "*", "/"]
              push!(node.children, generateNode(variables, minDepth-1,
                                                        depth-1, full))
          end
          return node
      else
          TreeNode(generateNumber(variables), Vector{TreeNode}())
      end
    end
end

function Tree(depth::Int, numVariables::Int, full::Bool)
    variables = generateVariables(numVariables)
    minDepth = 2
    head = generateNode(variables, minDepth, depth, full)
    return Tree(head, variables, 0.0, Float64[])
end

function assignValues!(variables::Vector{Variable}, values::Vector{Float64})
    if length(variables) != length(values)
        throw("amount of values should be same as of variables")
    end
    for i in 1:length(variables)
        variables[i].value = values[i]
    end
end

function computeFitness!(tree::Tree, X::Matrix{Float64}, y::Vector{Float64})
    fitness = 0.0
    num_rows = size(X, 1)
    tree.predictions = zeros(Float64, num_rows)

    try
        for i in 1:num_rows
            assignValues!(tree.variables, X[i, :])
            prediction = computeTree(tree)

            tree.predictions[i] = prediction
            fitness += abs(prediction - y[i])
        end
    catch e
        fitness = Inf
    end
    tree.fitness = fitness / num_rows
end

function Population(size::Int, numVariables::Int)
    population = Population(Vector{Tree}(), size)
    for _ in 1:size
        push!(population.individuals, Tree(rand(3:7), numVariables, rand(Bool)))
    end
    return population
end

function computePopulationFitness!(population::Population, X::Matrix{Float64},
                                                          y::Vector{Float64})
    for tree in population.individuals
        computeFitness!(tree, X, y)
    end
end

function best(population::Population)
    minimum(population.individuals)
end

function getAllNodeDescendants!(node::TreeNode, allNodes::Vector{TreeNode})
    if typeof(node.value) == Operation
        getAllNodeDescendants!(node.children[1], allNodes)
        if length(node.children) == 2
            getAllNodeDescendants!(node.children[2], allNodes)
        end
    end
    push!(allNodes, node)
end

function getAllTreeNodes(tree::Tree)
    allNodes = Vector{TreeNode}()
    getAllNodeDescendants!(tree.head, allNodes)
    return allNodes
end

function mutation!(tree::Tree, maximumDepth::Int)
      randNum = rand()
      allNodes = getAllTreeNodes(tree)

      if isempty(allNodes)
          return # Cannot mutate an empty tree
      end

      if randNum < 1/3
          operations = filter(x -> x.value isa Operation, allNodes)
          if !isempty(operations)
              node = rand(operations)
              node.value = generateOperation()
          else
              # Fallback to the following mutation option
              randNum += 1/3
          end
      end
      if 1/3 <= randNum < 2/3
          terminals = filter(x -> x.value isa Variable || x.value isa Float64, allNodes)
          if !isempty(terminals)
              node = rand(terminals)
              node.value = generateNumber(tree.variables)
          else
              # Fallback to the following mutation option
              randNum += 1/3
          end
      end
      if randNum >= 2/3
          if isempty(allNodes)
              return
          end
          node = rand(allNodes)
          depthPotential = maximumDepth - getNodeDepth(node, tree)
          minDepth = rand(0:max(0, min(1, depthPotential)))
          new_branch = generateNode(tree.variables, minDepth,
                          rand(minDepth:min(4, depthPotential)), rand(Bool))
          node.value = new_branch.value
          node.children = new_branch.children
      end
end

function getParent(treeNodes::Vector{TreeNode}, child::TreeNode)
    for node in treeNodes
        if child in node.children
            return (node, findfirst(==(child), node.children))
        end
    end
    return (nothing, nothing)
end

function getNodeDepth(target_node::TreeNode, current_head::TreeNode)
    if target_node === current_head
        return 0
    end

    if isempty(current_head.children)
        return -1 # Represents 'not found' in this path
    end

    for child in current_head.children
        child_depth = getNodeDepth(target_node, child)
        if child_depth != -1
            return 1 + child_depth
        end
    end

    return -1
end

function getNodeDepth(node::TreeNode, tree::Tree)
    getNodeDepth(node, tree.head)
end

function getNodeHeight(node::TreeNode)
    if isempty(node.children)
        0
    elseif length(node.children) == 1
        1 + getNodeHeight(node.children[1])
    else
        1 + max(getNodeHeight(node.children[1]), getNodeHeight(node.children[2]))
    end
end

function crossover!(tree1::Tree, tree2::Tree, maximumDepth::Int)
    allNodesTree1 = getAllTreeNodes(tree1)
    allNodesTree2 = getAllTreeNodes(tree2)

    properDepths = false
    crossoverNode1 = nothing
    crossoverNode2 = nothing

    while !properDepths
        crossoverNode1 = rand(allNodesTree1)
        crossoverNode2 = rand(allNodesTree2)

        height1 = getNodeHeight(crossoverNode1)
        height2 = getNodeHeight(crossoverNode2)

        # so we stay it permitted depth
        properDepths = (height1 + getNodeDepth(crossoverNode2, tree2) <= maximumDepth
                    &&  height2 + getNodeDepth(crossoverNode1, tree1) <= maximumDepth)
    end

    parent1, idx1 = getParent(allNodesTree1, crossoverNode1)
    parent2, idx2 = getParent(allNodesTree2, crossoverNode2)

    if isnothing(parent1) && crossoverNode1 != tree1.head ||
        isnothing(parent2) && crossoverNode2 != tree2.head
        throw("Error: Node without parent")
    end

    if crossoverNode1 == tree1.head && crossoverNode2 == tree2.head
        tree1.head = crossoverNode2
        tree2.head = crossoverNode1
    elseif crossoverNode1 == tree1.head
        tree1.head = crossoverNode2
        parent2.children[idx2] = crossoverNode1
    elseif crossoverNode2 == tree2.head
        tree2.head = crossoverNode1
        parent1.children[idx1] = crossoverNode2
    else
        parent1.children[idx1] = crossoverNode2
        parent2.children[idx2] = crossoverNode1
    end
end

function tournament_selection(population::Population)
    ind1, ind2, ind3 = sample(population.individuals, 3, replace=false)
    min(ind1, ind2, ind3)
end

function build_wheel(population::Population)
    fitnesses = [ind.fitness for ind in population.individuals]
    inverted = 1.0 ./ (fitnesses .+ 1e-6)
    probs = inverted ./ sum(inverted)
    return cumsum(probs)
end

function roulette_spin(population::Population, wheel::Vector{Float64})
    idx = min(searchsortedfirst(wheel, rand()), length(population.individuals))
    return population.individuals[idx]
end

function phenotypic_distance(preds1::Vector{Float64}, preds2::Vector{Float64})
    return sqrt(sum((preds1 .- preds2).^2))
end

function apply_fitness_sharing!(population::Population, sigma_share::Float64, alpha::Float64=1.0)
    N = length(population.individuals)
    m = zeros(Float64, N)

    for i in 1:N
        if population.individuals[i].fitness == Inf
            continue
        end

        for j in 1:N
            if population.individuals[j].fitness == Inf
                continue
            end

            d_ij = phenotypic_distance(
                population.individuals[i].predictions,
                population.individuals[j].predictions
            )

            # If distance is less than the threshold (sigma_share)
            if d_ij < sigma_share
                # sh(d) = 1 - (d / sigma_s)^alpha
                sh = 1.0 - (d_ij / sigma_share)^alpha
                m[i] += sh
            end
        end
    end

    for i in 1:N
        if population.individuals[i].fitness != Inf
            population.individuals[i].fitness *= m[i]
        end
    end
end

function compute_sigma_share(population::Population)
    all_preds = [ind.predictions for ind in population.individuals
                 if ind.fitness != Inf && !isempty(ind.predictions)]
    if isempty(all_preds)
        return 1.0
    end
    distances = [phenotypic_distance(all_preds[i], all_preds[j])
                 for i in 1:length(all_preds) for j in i+1:length(all_preds)]
    return isempty(distances) ? 1.0 : median(distances) * 0.5
end

function run_selection(population::Population, selection_method::Symbol, printable::Bool, niche::Bool,
    X::Matrix{Float64}, y::Vector{Float64}, generations::Int, maximumDepth::Int,
    mutation_rate::Float64, crossover_rate::Float64, elite_count::Int)

    best_fitness_hist = Float64[]

    # Initial fitness calculation
    computePopulationFitness!(population, X, y)

    for i in 0:generations

        # 1. EXTRACT ELITES (Using Raw Fitness)
        sorted = sort(population.individuals, by = x -> x.fitness)
        elites = [deepcopy(ind) for ind in sorted[1:elite_count]]

        best_ind = deepcopy(elites[1])
        current_fitness = best_ind.fitness

        push!(best_fitness_hist, current_fitness)
        if printable
            println("Generation $i. Best Fitness: $current_fitness")
        end

        if niche
            # 2. APPLY FITNESS SHARING (Penalizes the .fitness values)
            sigma_share = compute_sigma_share(population)
            apply_fitness_sharing!(population, sigma_share, 1.0)
        end


        # 3. SETUP SELECTION (Using the new Penalized Fitness)
        if selection_method == :roulette
            wheel = build_wheel(population)
            select = () -> roulette_spin(population, wheel)
        elseif selection_method == :tournament
            select = () -> tournament_selection(population)
        else
            throw(ArgumentError("Unknown selection method: $selection_method"))
        end


        # 4. GENERATE NEW POPULATION
        new_individuals = Vector{Tree}()

        while length(new_individuals) < population.size - elite_count
            randNum = rand()
            if randNum < crossover_rate
                # Crossover
                parent1 = select()
                parent2 = select()

                child1 = deepcopy(parent1)
                child2 = deepcopy(parent2)
                crossover!(child1, child2, maximumDepth)

                push!(new_individuals, child1)
                # Only push another if we have not exceeded population size
                if length(new_individuals) < population.size - elite_count
                    push!(new_individuals, child2)
                end
            elseif randNum < crossover_rate + mutation_rate
                # Mutation
                parent = select()
                child = deepcopy(parent)
                mutation!(child, maximumDepth)
                push!(new_individuals, child)
            else
                # Reproduction
                parent = select()
                child = deepcopy(parent)
                push!(new_individuals, child)
            end
        end

        # Re-evaluate New Generation
        # Elites go in unconditionally
        population.individuals = [elites; new_individuals]
        computePopulationFitness!(population, X, y)
    end
    return (best_fitness_hist, best(population))
end

function runSymbolicRegression(population::Population, X::Matrix{Float64}, y::Vector{Float64}, selection_method::Symbol, printable::Bool, niche::Bool, generations::Int,
    maximum_depth::Int, mutation_rate::Float64=0.25, crossover_rate::Float64=0.5, elite_count::Int = 3)

    name = niche ? "Niche " : ""
    if selection_method == :roulette
        name *= "Roulette"
    elseif selection_method == :tournament
        name *= "Tournament"
    else
        throw(ArgumentError("Unknown selection method: $selection_method"))
    end
    if printable
        println(name)
    end
    start_time = time_ns()
    history, best_tree = run_selection(population, selection_method, printable, niche,
                X, y, generations, maximum_depth, mutation_rate, crossover_rate, elite_count)
    elapsed = (time_ns() - start_time) / 1e9
    if printable
        println(name * "'s total time: ", round(elapsed, digits=2), " seconds")
        printBeautifulTree(best_tree)
        println("Heigh of tree:", getNodeHeight(best_tree.head))
    end
    return (history, best_tree, elapsed)
end

function doIterations(fileName, generations, populationSize, iterations, printable)
    X, y = prepare_data(fileName)
    individual_size = size(X, 2)
    mutation_rate = 0.25
    crossover_rate = 0.5
    maximum_depth = 8
    elite_count = 3

    absoluteBest = nothing
    fitness_hist_tournament = zeros(generations+1)
    fitness_hist_roulette = zeros(generations+1)
    fitness_hist_tournament_niche = zeros(generations+1)
    fitness_hist_roulette_niche = zeros(generations+1)

    avg_tournament_time = 0
    avg_roulette_time = 0
    avg_tournament_niche_time = 0
    avg_roulette_niche_time = 0

    for i in 1:iterations
        println("Iteration $i")
        population_roulette = Population(populationSize, individual_size)
        population_tournament = deepcopy(population_roulette)  # to have the same initial population
        population_tournament_niche = deepcopy(population_roulette)
        population_roulette_niche = deepcopy(population_roulette)
        
        tournament_hist, best_tournament_tree, tournament_time = runSymbolicRegression(population_tournament, X, y, :tournament, printable, false,
                                 generations, maximum_depth, mutation_rate, crossover_rate, elite_count)
        fitness_hist_tournament += tournament_hist
        avg_tournament_time += tournament_time
        
        roulette_hist, best_roulette_tree, roulette_time = runSymbolicRegression(population_roulette, X, y, :roulette, printable, false,
                                 generations, maximum_depth, mutation_rate, crossover_rate, elite_count)
        fitness_hist_roulette += roulette_hist
        avg_roulette_time += roulette_time

        tournament_niche_hist, best_tournament_niche_tree, tournament_niche_time = 
                            runSymbolicRegression(population_tournament_niche, X, y, :tournament, printable, true,
                                 generations, maximum_depth, mutation_rate, crossover_rate, elite_count)
        fitness_hist_tournament_niche += tournament_niche_hist
        avg_tournament_niche_time += tournament_niche_time
        roulette_niche_hist, best_roulette_niche_tree, roulette_niche_time = 
                            runSymbolicRegression(population_roulette_niche, X, y, :roulette, printable, true,
                                 generations, maximum_depth, mutation_rate, crossover_rate, elite_count)                         
        fitness_hist_roulette_niche += roulette_niche_hist
        avg_roulette_niche_time += roulette_niche_time

        theBest = min(best_tournament_tree, best_roulette_tree, best_tournament_niche_tree, best_roulette_niche_tree)
        if !isnothing(absoluteBest)
            if theBest < absoluteBest
                absoluteBest = deepcopy(theBest)
            end
        else
            absoluteBest = deepcopy(theBest)
        end

        if printable
            p = plot(0:generations, [tournament_hist, roulette_hist, tournament_niche_hist, roulette_niche_hist], yaxis=:log,
                title="Error history",
                xlabel="Generation",
                ylabel="Mean Absolute Error",
                label=["Tournament Selection" "Roulette Selection" "Niche Tournament Selection" "Niche Roulette Selection"],
                linewidth=2,
                legend=:bottomleft)

            savefig(p, fileName*"_fitness_evolution_$i.png")
            display(p)

            tournament_values = Float64[]
            roulette_values = Float64[]
            tournament_niche_values = Float64[]
            roulette_niche_values = Float64[]
            for j in 1:size(X, 1)
                assignValues!(best_tournament_tree.variables, X[j, :])
                push!(tournament_values, computeTree(best_tournament_tree))

                assignValues!(best_roulette_tree.variables, X[j, :])
                push!(roulette_values, computeTree(best_roulette_tree))

                assignValues!(best_tournament_niche_tree.variables, X[j, :])
                push!(tournament_niche_values, computeTree(best_tournament_niche_tree))

                assignValues!(best_roulette_niche_tree.variables, X[j, :])
                push!(roulette_niche_values, computeTree(best_roulette_niche_tree))
            end

            p = plot(1:size(X, 1), [y, tournament_values, roulette_values, tournament_niche_values, roulette_niche_values], #=xaxis=:log,=#
                title="Selection comparison",
                xlabel="Number",
                ylabel="Value of function",
                label=["Actual value" "Tournament Selection" "Roulette Selection" "Niche Tournament Selection" "Niche Roulette Selection"],
                linewidth=2,
                legend=:topleft)

            savefig(p, fileName*"_approximation_$i.png")
            display(p)
        end
    end

    fitness_hist_tournament /= iterations
    fitness_hist_roulette /= iterations
    fitness_hist_tournament_niche /= iterations
    fitness_hist_roulette_niche /= iterations

    avg_tournament_time /= iterations
    avg_roulette_time /= iterations
    avg_tournament_niche_time /= iterations
    avg_roulette_niche_time /= iterations

    println("The Best Solution:")
    printBeautifulTree(absoluteBest)
    computeFitness!(absoluteBest, X, y)
    println("It's fitness: $(absoluteBest.fitness)")

    println("Average Tournament time: $avg_tournament_time")
    println("Average Roulette time: $avg_roulette_time")
    println("Average Niche Tournament time: $avg_tournament_niche_time")
    println("Average Niche Roulette time: $avg_roulette_niche_time")

    best_tree_values = Float64[]
    for i in 1:size(X, 1)
        assignValues!(absoluteBest.variables, X[i, :])
        push!(best_tree_values, computeTree(absoluteBest))
    end

    p = plot(1:size(X, 1), [y, best_tree_values], #=xaxis=:log,=#
        title="The Best Approximation comparison",
        xlabel="Number",
        ylabel="Value of function",
        label=["Actual value" "The Best Approximation"],
        linewidth=2,
        legend=:topleft)

    savefig(p, fileName*"_best_approximation.png")
    display(p)

    p = plot(0:generations, [fitness_hist_tournament, fitness_hist_roulette, fitness_hist_tournament_niche, fitness_hist_roulette_niche], yaxis=:log,
        title="Comparison",
        xlabel="Generation",
        ylabel="Average MAE after $iterations iterations",
        label=["Tournament Selection" "Roulette Selection" "Niche Tournament Selection" "Niche Roulette Selection"],
        linewidth=2,
        legend=:bottomleft)

    savefig(p, fileName*"_average_fitness_evolution.png")
    display(p)
end

function main()
    start_time = time_ns()
    doIterations("pi.csv", 35, 150, 10, false)
    elapsed = (time_ns() - start_time) / 1e9
    println("Total execution time: ", round(elapsed, digits=2), " seconds")
end

main()