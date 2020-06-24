#!/bin/bash
source .env
echo AtulizandoBD
python3 ./scripts/create_db.py
echo Treinando
python3 ./scripts/n4j_node2Vec.py
echo Finalizado
