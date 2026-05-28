from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    model_name: str = "gpt-4o"
    data_dir: str = "data/raw"


settings = Settings()
