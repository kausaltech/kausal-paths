# Kausal Paths

Kausal Paths is a tool for predicting the future emissions of cities based on historical emission data and various climate actions. Users can compare how emissions develop in different scenarios. Data is provided using a GraphQL API.

## Installation
### Prerequisites
Make sure you have installed the following:
- a package manager (e.g., homebrew for macOS, apt for Ubuntu, etc.)
- git
- python
- uv (python package manager)
- direnv (auto-loads environment variables)
        Beginner Hints: before direnv works you need to hook it into your shell: depending on which shell you're using (find out with echo $SHELL)
        add eval "$(direnv hook bash)" to your ~/.bashrc OR eval "$(direnv hook zsh)" to your ~/.zshrc (create those files if they don't exist yet).
        After that, restart the terminal or run source ~/.bashrc OR source ~/.zshrc.


### Development
After cloning the repository, move to your project root directory (paths folder) and allow loading environment variables:
`direnv allow`

In the project root directory, create and activate a Python virtual environment:

```shell
uv venv
source .venv/bin/activate
```

Install the required Python packages:

```shell
uv sync
```


Configure the PyPI index URL in your `.envrc` file (create it if it doesn't exist yet) if you have access to the Kausal private extensions (ask a team member for access):

```shell
export UV_INDEX_KAUSAL_USERNAME=...
export UV_INDEX_KAUSAL_PASSWORD=...
```


Then install the dependencies like this:

```shell
uv sync --extra kausal
```

If you need to run Jupyter notebooks, include the `notebook` dependency group:

```shell
uv sync --group notebook --extra kausal
```

Create a file called `local_settings.py` in your repository root with the following contents:

```python
from paths.settings import BASE_DIR

DEBUG = True

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'paths',
        'ATOMIC_REQUESTS': True,
    }
}
```

> _Note for macOS users: You might need to install a few packages separately_
>
> ```
> brew install snappy
> CPPFLAGS="-I/opt/homebrew/include -L/opt/homebrew/lib" pip install python-snappy
> ```
> 
> ```
> brew install postgresql@16
> brew services start postgresql@16
> ```
>
> ```
> cd ~/Documents/kausal-paths  # or wherever your project is
> source .venv/bin/activate
> uv pip install psycopg2
> ```
>
> ```
> brew install gdal geos
> ```
> then set the library path in your local kausal-paths/local_settings.py to the correct path shown in `ls -la /opt/homebrew/lib/*gdal*`:
> ```
> GDAL_LIBRARY_PATH = '/opt/homebrew/lib/libgdal.dylib'
> GEOS_LIBRARY_PATH = '/opt/homebrew/lib/libgeos_c.dylib' 
> ```
>
> ```
> brew tap hashicorp/tap
> brew install hashicorp/tap/vault
> ```



Run migrations:

```shell
python manage.py migrate
```

Create a superuser:

> You might need the following translations during the createsuperuser operation: käyttäjätunnus = username, sähköpostiosoite = e-mail

```shell
python manage.py createsuperuser
```

Compile the translation files:

```shell
python manage.py compilemessages
```



Set Credentials and update database:

First, make sure your team added your access rights in Vault keycloak and send you the command to login again. 

Clone the scripts repository (ask a team mate where to find it).

Go to paths root directory and run the script `../scripts/common/switch-region.sh` while in paths root directory, which should automatically update all needed credentials in `.secrets/db-backup`

Follow the instructions the command gives you, there's a variable you need to put in .env and then the commands to restore the db 

Then run

```shell
dropdb paths ; createdb paths && DB_BACKUP_SECRET_PATH=.secrets/db-backup kausal_common/docker/manage-db-backup.sh restore
```


You can now run the backend:

```shell
python manage.py runserver
```

The GraphQL API is now available at `http://127.0.0.1:8000/v1/graphql/`.
