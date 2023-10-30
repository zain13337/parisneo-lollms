# lollms-settings

## Description
The `lollms-settings` tool is used to configure multiple aspects of the lollms project. Lollms is a multi-bindings LLM service that serves multiple LLM models that can generate text out of a prompt.

## Usage
To use the `lollms-settings` tool, you can run the following command:

```
python lollms_settings.py [--configuration_path CONFIGURATION_PATH] [--reset_personal_path] [--reset_config] [--reset_installs] [--default_cfg_path DEFAULT_CFG_PATH] [--tool_prefix TOOL_PREFIX] [--set_personal_folder_path SET_PERSONAL_FOLDER_PATH] [--install_binding INSTALL_BINDING] [--install_model INSTALL_MODEL] [--select_model SELECT_MODEL] [--mount_personalities MOUNT_PERSONALITIES] [--set_personal_foldrer SET_PERSONAL_FOLDRE] [--silent]
```

### Arguments

- `--configuration_path`: Path to the configuration file.
- `--reset_personal_path`: Reset the personal path.
- `--reset_config`: Reset the configurations.
- `--reset_installs`: Reset all installation status.
- `--default_cfg_path`: Reset all installation status.
- `--tool_prefix`: A prefix to define what tool is being used (default `lollms_server_`).
  - lollms applications prefixes:
    - lollms server: `lollms_server_`
    - lollms-elf: `lollms_elf_`
    - lollms-webui: `""`
    - lollms-discord-bot: `lollms_discord_`
- `--set_personal_folder_path`: Forces installing and selecting a specific binding.
- `--install_binding`: Forces installing and selecting a specific binding.
- `--install_model`: Forces installing and selecting a specific model.
- `--select_model`: Forces selecting a specific model.
- `--mount_personalities`: Forces mounting a list of personas.
- `--set_personal_foldrer`: Forces setting personal folder to a specific value.
- `--silent`: This will operate in silent mode, no menu will be shown.

### Examples
Here are some examples of how to use the `lollms-settings` tool:

1. Reset the configurations:
```
python lollms_settings.py --reset_config
```

2. Install and select a specific binding:
```
python lollms_settings.py --install_binding <binding_name>
```

3. Install and select a specific model:
```
python lollms_settings.py --install_model <model_path>
```

4. Select a specific model:
```
python lollms_settings.py --select_model <model_name>
```

5. Mount a list of personas:
```
python lollms_settings.py --mount_personalities <persona1> <persona2> ...
```

6. Set personal folder to a specific value:
```
python lollms_settings.py --set_personal_foldrer <folder_path>
```

7. Run in silent mode:
```
python lollms_settings.py --silent
```

## License
This project is licensed under the Apache 2.0 License.