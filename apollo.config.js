module.exports = {
  client: {
    service: {
      name: 'paths',
      url: 'http://localhost:8000/v1/graphql/',
    },
    includes: ['**/*.gql'],
    excludes: ['.*'],
  }
};
