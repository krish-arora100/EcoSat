from sentinelhub import SHConfig

config = SHConfig()

print(config)

print(SHConfig.get_config_location())

config.sh_client_id = "INSERT_KEY"
config.sh_client_secret = "INSERT_KEY"
config.instance_id = "INSERT_KEY"

config.save()

if not config.sh_client_id or not config.sh_client_secret:
    print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")
