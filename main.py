from fastapi import FastAPI

app = FastAPI(name="langgraph-ai-agent")


@app.get("/health")
async def health_check():
    return {"status":"ok"}


def main():
    print("Hello from new-projectpart2!")


if __name__ == "__main__":
    main()
