import torch

inputsList = [
    [0.43, 0.15, 0.89], #your
    [0.55, 0.87, 0.66], #journey
    [0.57, 0.85, 0.64], #starts
    [0.22, 0.58, 0.33], #with
    [0.77, 0.25, 0.10], #one
    [0.05, 0.80, 0.55], #step
]

def calculateAttentionScores(inputs, queryId):
    query = inputs[queryId]
    attentionScores = torch.empty(inputs.shape[0])

    print(f"Query: {query}")
    print(f"Query Shape: {query.shape}")

    for index, input in enumerate(inputs):
        print(f"index[{index}] = {input}, shape: {input.shape}")
        attentionScores[index] = torch.dot(query, input)

    print(f"Attention scores: {attentionScores}")
    return attentionScores

def normalizeUsingSoftmax(attentionScores):
    expScores = torch.exp(attentionScores)
    sumExpScores = torch.sum(expScores)
    return expScores / sumExpScores

def getContextVector(inputs, attentionScores, queryId):
    query = inputs[queryId]
    contextVector = torch.zeros(query.shape)
    for index, input in enumerate(inputs):
        contextVector += attentionScores[index] * input
    return contextVector

def basicSelfAttentionTest():
    print(f"***Starting test - basicSelfAttentionTest***")
    inputs = torch.tensor(inputsList)

    attentionScores = calculateAttentionScores(inputs, 1)
    normalizedAttentionScores = normalizeUsingSoftmax(attentionScores)
    print(f"Normalized attention scores: {normalizedAttentionScores}")
    print(f"Sum of normalized attention scores: {torch.sum(normalizedAttentionScores)}")

    normalizedWithPytorchSoftmax = torch.softmax(attentionScores, dim=0)
    print(f"Normalized attention scores using PyTorch softmax: {normalizedWithPytorchSoftmax}")
    print(f"Sum of normalized attention scores using PyTorch softmax: {torch.sum(normalizedWithPytorchSoftmax)}")

    contextVector = getContextVector(inputs, normalizedWithPytorchSoftmax, queryId=1)

    print(f"Context vector: {contextVector}")
    print(f"Original input: {inputs[1]}")

    print(f"***End test - basicSelfAttentionTest***")

def fullSelfAttentionTest():
    print(f"***Starting test - fullSelfAttentionTest***")
    inputs = torch.tensor(inputsList)

    fullAttentionScores = torch.empty((inputs.shape[0], inputs.shape[0]))
    print(f"Full attention scores shape: {fullAttentionScores.shape}, inputs shape: {inputs.shape}")

    # @ is the matrix multiplication operator in python
    # This was introduced in python3.5.
    fullAttentionScores = inputs @ inputs.T

    print(f"Full attention scores: {fullAttentionScores}")
    normalizedAttentionScores = torch.softmax(fullAttentionScores, dim=1)
    print(f"Normalized attention scores: {normalizedAttentionScores}")
    print(f"Sum of normalized attention scores: {torch.sum(normalizedAttentionScores, dim=1)}")

    print(f"normalizedAttentionScores shape: {normalizedAttentionScores.shape}")
    print(f"inputs shape: {inputs.shape}")

    contextVector = normalizedAttentionScores @ inputs
    print(f"Context vector: {contextVector}")

    print(f"***End test - fullSelfAttentionTest***")

def selfAttentionWithKeyValueTest():
    print(f"***Starting test - selfAttentionWithKeyValueTest***")
    inputs = torch.tensor(inputsList)

    queryIndex = 1
    xAtIndex = inputs[queryIndex]
    dIn = inputs.shape[1]
    dOut = 2

    torch.manual_seed(123)
    wQuery = torch.nn.Parameter(torch.rand(dIn, dOut), requires_grad=False)
    wKey = torch.nn.Parameter(torch.rand(dIn, dOut), requires_grad=False)
    wValue = torch.nn.Parameter(torch.rand(dIn, dOut), requires_grad=False)

    query = xAtIndex @ wQuery
    keys = inputs @ wKey
    values = inputs @ wValue

    key = keys[queryIndex]

    print(f"key shape: {key.shape}, query shape: {query.shape}")

    attentionScores = query @ keys.T

    attentionWeights = torch.softmax(attentionScores / dOut ** 0.5, dim=0)

    contextVector = attentionWeights @ values

    print(f"query: {query}")
    print(f"keys: {keys}")
    print(f"values: {values}")
    print(f"key: {key}")
    print(f"attentionScore: {attentionScores}")
    print(f"attentionWeights: {attentionWeights}")
    print(f"contextVector: {contextVector}")
    print(f"dK = {keys.shape[-1]}, keys shape = {keys.shape}")

    print(f"***End test - selfAttentionWithKeyValueTest***")

def main():
    #basicSelfAttentionTest()
    #fullSelfAttentionTest()
    selfAttentionWithKeyValueTest()

if __name__ == "__main__":
    main()

