from py2neo import Graph
from dotenv import load_dotenv
import os
load_dotenv()


def test_connection():
    graph = Graph(os.getenv('N4J_URL'), auth=(
        os.getenv('N4J_USER'), os.getenv('N4J_PASS')))
    result = graph.run("Match () Return 1 Limit 1")
