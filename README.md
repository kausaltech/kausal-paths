# Kausal Paths

Kausal Paths is a tool for predicting the future emissions of cities based on historical emission data and various climate actions. Users can compare how emissions develop in different scenarios. Data is provided using a GraphQL API.

## Installation

### Development

In the project root directory, create and activate a Python virtual environment:

```shell
python3 -m venv venv
source venv/bin/activate
```

Install the required Python packages:

```shell
pip install -r requirements.txt
```

> _Note for macOS users: If you run into issues installing python-snappy, install it separately first_
>
> ```
> brew install snappy
> CPPFLAGS="-I/opt/homebrew/include -L/opt/homebrew/lib" pip install python-snappy
> ```

Create a file called `local_settings.py` in your repo root with the following contents:

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

You can now run the backend:

```shell
python manage.py runserver
```

The GraphQL API is now available at `http://127.0.0.1:8000/v1/graphql/`.
