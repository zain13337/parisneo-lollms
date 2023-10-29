# lollms-settings

The lollms-settings app is used to configure multiple aspects of the lollms project. Lollms is a multi bindings LLM service that serves multiple LLM models that can generate text out of a prompt.

## Usage

To use the lollms-settings app, you need to provide command-line arguments. Here are the available arguments:

### `--configuration_path`

- Description: Path to the configuration file.
- Default: None

### `--reset_personal_path`

- Description: Reset the personal path.
- Action: Store True

### `--reset_config`

- Description: Reset the configurations.
- Action: Store True

### `--reset_installs`

- Description: Reset all installation status.
- Action: Store True

### `--default_cfg_path`

- Description: Reset all installation status.
- Type: String
- Default: None

### `--tool_prefix`

- Description: A prefix to define what tool is being used.
- Type: String
- Default: "lollms_server_"

### `--set_personal_folder_path`

- Description: Forces installing and selecting a specific binding.
- Type: String
- Default: None

### `--install_binding`

- Description: Forces installing and selecting a specific binding.
- Type: String
- Default: None

### `--install_model`

- Description: Forces installing and selecting a specific model.
- Type: String
- Default: None

### `--select_model`

- Description: Forces selecting a specific model.
- Type: String
- Default: None

### `--mount_personalities`

- Description: Forces mounting a list of personas.
- Type: List of Strings
- Default: None

### `--set_personal_foldrer`

- Description: Forces setting personal folder to a specific value.
- Type: String
- Default: None

### `--silent`

- Description: This will operate in silent mode, no menu will be shown.
- Action: Store True

## Example

To run the lollms-settings app with specific configurations, you can use the command-line arguments. Here's an example:

```bash
python lollms-settings.py --configuration_path /path/to/configuration/file --reset_personal_path --install_binding binding_name
```

In this example, the `--configuration_path` argument is used to specify the path to the configuration file. The `--reset_personal_path` argument is used to reset the personal path, and the `--install_binding` argument is used to force installing and selecting a specific binding.

## Additional Notes

- Make sure to provide the necessary permissions and access rights to the configuration file and any other files or directories required by the lollms-settings app.
- Refer to the lollms documentation for more information on configuring and using the lollms project.
