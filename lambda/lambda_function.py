import json
import torch


def lambda_handler(event, context):
    # Load the data from the request
    data = json.loads(event['body'])
    print(data)
    # Convert the data to a PyTorch tensor
    t = torch.tensor(data.input)
    print(t)

    # TODO: Load the reinforcement learning model

    # TODO: Get the predicted action from the model

    # Construct the response object
    response = {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json'
        },
        'body': json.dumps({'action': "hello"})
    }

    return response
