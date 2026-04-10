# import Pkg; Pkg.add("Latexify"); Pkg.add("StatsBase"); Pkg.add("CSV"); Pkg.add("DataFrames")  # for first time run
using Latexify
using Random
using StatsBase: sample
using CSV
using DataFrames

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
        sqrt(operand1)
    elseif operation.name == "log"
        if operand1 == 0
            throw("log zero")
        end
        log(operand1)
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
            if operand2 == 0
                throw("Division by zero")
            end
            operand1 / operand2
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
    if randNum < 1/8
        Operation("sqrt")
    elseif randNum < 1/4
        Operation("log")
    elseif randNum < 3/8
        Operation("floor")
    elseif randNum < 1/2
        Operation("ceil")
    elseif randNum < 5/8
        Operation("+")
    elseif randNum < 3/4
        Operation("-")
    elseif randNum < 7/8
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
    alphabetVector = Vector{}()
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
        TreeNode(generateNumber(variables), Vector{}())
    else
      if rand() < 0.5 || full || minDepth > 0
          operation = generateOperation()
          node = TreeNode(operation, Vector{}())
          child1 = generateNode(variables, minDepth-1, depth-1, full)
          node.children = [child1]
          if operation.name in ["+", "-", "*", "/"]
              push!(node.children, generateNode(variables, minDepth-1,
                                                        depth-1, full))
          end
          return node
      else
          TreeNode(generateNumber(variables), Vector{}())
      end
    end
end

function Tree(depth::Int, numVariables::Int, full::Bool)
    variables = generateVariables(numVariables)
    minDepth = 3
    head = generateNode(variables, minDepth, depth, full)
    return Tree(head, variables, 0)
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
    squared_errors_sum = 0.0
    fitness = Inf
    try
        for i in 1:X.size[1]
            assignValues!(tree.variables, X[i, :])
            prediction = computeTree(tree)
            squared_errors_sum += (prediction - y[i])^2
        end
        fitness = squared_errors_sum / X.size[1]
    catch e
        fitness = Inf
    end
    tree.fitness = fitness
end

function Population(size::Int, numVariables::Int)
    population = Population(Vector{}(), size)
    for i in 1:size
        randNum = rand()
        if randNum < 0.1
            push!(population.individuals, Tree(3, numVariables, true))
        elseif randNum < 0.2
            push!(population.individuals, Tree(4, numVariables, true))
        elseif randNum < 0.3
            push!(population.individuals, Tree(5, numVariables, true))
        elseif randNum < 0.4
            push!(population.individuals, Tree(6, numVariables, true))
        elseif randNum < 0.5
            push!(population.individuals, Tree(7, numVariables, true))
        elseif randNum < 0.6
            push!(population.individuals, Tree(3, numVariables, false))
        elseif randNum < 0.7
            push!(population.individuals, Tree(4, numVariables, false))
        elseif randNum < 0.8
            push!(population.individuals, Tree(5, numVariables, false))
        elseif randNum < 0.9
            push!(population.individuals, Tree(6, numVariables, false))
        else
            push!(population.individuals, Tree(7, numVariables, false))
        end
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
      if randNum < 2/3
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
          node = rand(allNodes)
          height = getNodeHeight(node)
          minDepth = rand(0:max(0, min(1, height)))
          new_branch = generateNode(tree.variables, minDepth, rand(minDepth:min(4, height)), rand(Bool))
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
        return Inf # Represents 'not found' in this path
    end

    for child in current_head.children
        child_depth = getNodeDepth(target_node, child)
        if child_depth != Inf
            return 1 + child_depth
        end
    end

    return Inf
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

function tournament(tree1::Tree, tree2::Tree, tree3::Tree)
    minimum(tree1, tree2, tree3)
end



X, y = prepare_data("pi.csv")

population = Population(100, 1)
computePopulationFitness!(population, X, y)
bestTree = best(population)

println(stringTree(bestTree))
println(computeFitness(bestTree, X, y))