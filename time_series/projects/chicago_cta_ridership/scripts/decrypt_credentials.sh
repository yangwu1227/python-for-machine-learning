#!/bin/bash

# Prompt the user for the decryption passphrase
read -sp "Enter decryption passphrase: " passphrase

if ! command -v openssl &> /dev/null; then
    echo "openssl is not found. Installing it using brew..."
    brew install openssl
fi

# Decrypt the credentials.json.enc file using the provided passphrase
output=$(echo $passphrase | openssl aes-256-cbc -d -a -salt -pbkdf2 -in credentials.json.enc -pass stdin)

echo "Decryption complete."

if ! command -v jq &> /dev/null; then
    echo "jq is not found. Installing it using brew..."
    brew install jq
fi

# Parse the JSON output to get the access key, secret key, and session token
aws_access_key_id=$(echo "$output" | jq -r '.Credentials.AccessKeyId')
aws_secret_access_key=$(echo "$output" | jq -r '.Credentials.SecretAccessKey')
aws_session_token=$(echo "$output" | jq -r '.Credentials.SessionToken')

# Prompt the user for confirmation to save the credentials to a .env file
read -p "Do you want to save the credentials to a .env file in the current directory? (yes/no) " response

response=$(echo "$response" | tr '[:upper:]' '[:lower:]')

if [[ $response == "yes" || $response == "y" ]]; then
    # Write the credentials to a .env file
    echo "AWS_ACCESS_KEY_ID=$aws_access_key_id" > .env
    echo "AWS_SECRET_ACCESS_KEY=$aws_secret_access_key" >> .env
    echo -n "AWS_SESSION_TOKEN=$aws_session_token" >> .env

    echo ".env file created successfully."
elif [[ $response == "no" || $response == "n" ]]; then
    read -p "Please provide the root project path: " project_root

    # Check if the provided path exists and is a directory
    if [[ -d "$project_root" ]]; then
        # Change to the provided directory
        cd "$project_root"

        # Write the credentials to a .env file in the provided directory
        echo "AWS_ACCESS_KEY_ID=$aws_access_key_id" > .env
        echo "AWS_SECRET_ACCESS_KEY=$aws_secret_access_key" >> .env
        echo -n "AWS_SESSION_TOKEN=$aws_session_token" >> .env 

        echo ".env file created in $project_root."
    else
        echo "Provided path does not exist or is not a directory. Exiting."
        exit 1
    fi
else
    echo "Invalid response. Exiting."
    exit 1
fi

# Cleanup
unset passphrase
rm credentials.json.enc