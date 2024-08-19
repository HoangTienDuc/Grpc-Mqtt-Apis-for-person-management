#######################################################
# add_update users
## Input
{
    "id": "123456",
    "command": "FR_ADD_UPDATE_USER",
    "name": "TienD",
    "uuid": "123456",
    "organization": "organization",
    "time_start": 0, # Timestamp
    "time_end": 0
}

## Output
{
    "id": "123456",
    "command": "FR_ADD_UPDATE_USER",
    "status": True,
    "content": 4,
    "name": "TienD",
    "uuid": "123456",
    "organization": "organization",
    "time_start": 0, # Timestamp
    "time_end": 0
}

#######################################################

# delete users
## Input
{
    "id": "123456",
    "command": "FR_DELETE_USER",
    "name": "TienD",
    "uuid": "123456",
    "organization": "organization"
}


## Output
{
    "id": "123456",
    "command": "FR_DELETE_USER",
    "status": True,
    "content": 4,
    "name": "TienD",
    "uuid": "123456",
    "organization": "organization"
}
#######################################################

# change user name
## Input
{
    "id": "123456",
    "command": "FR_CHANGE_USER_NAME",
    "old_name": "TienD",
    "new_name": "Tien Duc",
    "uuid": "123456",
    "organization": "organization",
    "new_time_start": 0, # Timestamp
    "new_time_end": 0
}


## Output
{
    "id": "123456",
    "command": "FR_CHANGE_USER_NAME",
    "status": True,
    "content": 4,
    "old_name": "TienD",
    "new_name": "Tien Duc",
    "uuid": "123456",
    "organization": "organization",
    "new_time_start": 0, # Timestamp
    "new_time_end": 0
}

#######################################################
# detect faces
## Input
{
    "id": "123456",
    "command": "FR_DETECT_FACE",
    
    "images": [
        "base64_image_1", # size 112x112
        "base64_image_2", # size 112x112
        ...
    ]
}


## Output
{
    "id": "123456",
    "command": "FR_DETECT_FACE",
    "status": True,
    "content": 4,
}

#######################################################
# recognize
## Input
{
    "id": "123456",
    "command": "RECOGNIZE",
    "organization": "organization",
    "is_recongize_safety": True,
    "images": [
        "base64_image_1",
        "base64_image_2",
        ...
    ]
}


## Output
{
    "id": "123456",
    "command": "RECOGNIZE",
    "organization": "organization",
}

#######################################################
# Create classifier organization
## Input
{
    "id": "123456",
    "command": "CREATE_ORGANIZATION",
    "organization": "organization1",
}


## Output
{
    "id": "123456",
    "command": "CREATE_ORGANIZATION",
    "status": True,
    "content": None,
    "organization": "organization1",
}

#######################################################
# Active classifier organization
## Input
{
    "id": "123456",
    "command": "ACTIVE_ORGANIZATION",
    "organization": "organization1",
}


## Output
{
    "id": "123456",
    "command": "ACTIVE_ORGANIZATION",
    "status": True,
    "content": None,
    "organization": "organization1",
}
#######################################################
# Deactive classifier organization
## Input
{
    "id": "123456",
    "command": "DEACTIVATE_ORGANIZATION",
    "organization": "organization1",
}


## Output
{
    "id": "123456",
    "command": "DEACTIVATE_ORGANIZATION",
    "status": True,
    "content": None,
    "organization": "organization1",
}
#######################################################
# Remove classifier organization
## Input
{
    "id": "123456",
    "command": "REMOVE_ORGANIZATION",
    "organization": "organization1",
}


## Output
{
    "id": "123456",
    "command": "REMOVE_ORGANIZATION",
    "status": True,
    "content": None,
    "organization": "organization1",
}
#######################################################
