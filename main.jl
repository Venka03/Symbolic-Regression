# import Pkg; Pkg.add("Latexify"); Pkg.add("StatsBase")  # for first time run
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
    variable::Vector{Variable}
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

function printBeatifulTree(tree::Tree)
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
    elseif operand2 != nothing
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
    if randNum < 1/3  # generate float from 0 to 10
        10*rand()
    elseif randNum < 2/3  # generate int from 0 to 10
        float(rand(0:10))
    else
        rand(variables)  # take randomly variable
    end
end

function generateVariables(num::Int)
    alphabet = "abcdefghijklmnopqrtsuvwxyz"
    alphabetVector = Vector{}()
    for el in alphabet
        push!(alphabetVector, string(el))
    end
    variableNames = sample(alphabetVector, num, replace=false)

    variables = Vector{Variable}()

    for name in variableNames
        push!(variables, Variable(name, 0))  # make 0 value as default
    end
    return variables
end

function generateNode(variables::Vector{Variable}, depth::Int)
    if depth == 1
        TreeNode(generateNumber(variables), Vector{}())
    else
        randNum = rand()
        if randNum < 0.5
            operation = generateOperation()
            node = TreeNode(operation, Vector{}())
            child1 = generateNode(variables, depth-1)
            node.children = [child1]
            if operation.name in ["+", "-", "*", "/"]
                push!(node.children, generateNode(variables, depth-1))
            end
            return node
        else
            TreeNode(generateNumber(variables), Vector{}())
      end
    end
end

function generateTree(depth::Int, numVariables::Int)
    variables = generateVariables(numVariables)
    head = generateNode(variables, depth)
    return Tree(head, variables)
end

tree = generateTree(5, 1)

println(stringTree(tree))
println(computeTree(tree))