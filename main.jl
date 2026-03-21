# import Pkg; Pkg.add("Latexify")  # for first time run
using Latexify

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
        return node.value.name
    end
    if typeof(node.value) != Operation
        return string(node.value)
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

var = Variable("x", 5)
a = TreeNode(2.0, Vector{}())
d = TreeNode(var, Vector{}())
e = TreeNode(Operation("*"), [a, d])
c = TreeNode(4.0, Vector{}())
b = TreeNode(Operation("+"), [e, c])
f = TreeNode(Operation("sqrt"), [b])

tree = Tree(f, var)

printBeatifulTree(tree)
println(computeTree(tree))