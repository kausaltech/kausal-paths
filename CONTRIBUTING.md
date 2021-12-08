# Contributing to Kausal Paths

We welcome open-source contributions to the project. This document should get you started, but feel free to contact us on GitHub if there are any questions.

## Development environment

For setting up a development instance of the software, see the [README.md](https://github.com/kausaltech/kausal-paths/blob/main/README.md) file. You will need Python 3 and it's probably wise to use Linux and PostgreSQL for development.

## Reporting bugs

If you think you found a bug, please report it using our [GitHub issue tracker](https://github.com/kausaltech/kausal-paths/issues/) and include all necessary information for reproducing it.

## Contributing code

If you would like to contribute some code, please create a pull request on GitHub. Before you do, make that all unit tests succeed.

You can run the unit tests like this:

```
pytest --reuse-db
```

For contributions that are not straightforward, please include unit tests in your pull request to check if your components have the desired behavior.

We ask you to follow the same coding style as the rest of the code. We try to follow [PEP 8](https://www.python.org/dev/peps/pep-0008/). Please also make sure there are no linting errors.
