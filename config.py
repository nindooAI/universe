import os 

config = {
    'neo4j': {
        'address': os.getenv('N4J_URL'), 
        'user': os.getenv('N4J_USER'),
        'password': os.getenv('N4J_PASS')
    },
    'crawler': { 
        'address': os.getenv('CRAWLER_URL')
    }

}