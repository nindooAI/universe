# Universe
Sistema de recomendação em grafos da Nindoo


## Instalação

Para instalar rode os seguintes comandos:

```shell
git clone https://github.com/nindooAI/universe

```
## Executando

Um exemplo de `.env`: encontra-se em `env.sample`.

### Docker
Para rodar a API a apartir do Docker, execute o seguinte comando, a partir da pasta `universe`:
```
sudo docker build -t universe-api:v2 .
sudo docker run -p 8000:8000 --env-file ~/.env universe-api:v2
```
