import fs from "fs";

const LOCAL_SCHEMA_FILE = "./__generated__/schema.graphql";

const urlService = {
  url: "http://localhost:8000/v1/graphql/",
};
const localFileService = {
  localSchemaFile: LOCAL_SCHEMA_FILE,
};

const service = {
  name: "paths",
  ...(fs.existsSync(LOCAL_SCHEMA_FILE) ? localFileService : urlService),
};

export default {
  client: {
    service,
    includes: ["**/*.gql", "**/tests/*.py"],
    excludes: [".*"],
  },
};
