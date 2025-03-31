# visualizer-bke

## Extracting data from datafed
### If you have data locally you can copy the folder path and use it later for generating embedding
### Else you extract from datafed using 
- - adjust env according to your datafed configuration
- - **Must have a globus container running**
- - if you dont have globus container running you can checkout this 
- - then run the following 
```bash
python datafed_extractor.py
``` 

## Initialize docker container for mongoDb by 
```bash
docker run --name mongodb \
  -p 27017:27017 \
  --restart unless-stopped \
  -d mongodb/mongodb-community-server:latest

```
## Create a user
```bash
mongosh
```
```bash
use admin
db.createUser({
  user: "mongouser",
  pwd: "password123",
  roles: [{ role: "readWrite", db: "*" }]
})

```

## Install python dependencies 
```bash
pip install -r requirements.txt
```

## Run the backend server locally 
```bash
python app.py
```

