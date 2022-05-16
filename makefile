docker-build:
	docker build -t khmer-tts-backend-fastapi -f Dockerfile .

docker-run:
	docker run -d -p 8000:8000 --name khmer-tts-backend-fastapi khmer-tts-backend-fastapi

docker-up:
	make docker-build
	make docker-run

dev:
	docker compose -f dev.compose.yml up --build