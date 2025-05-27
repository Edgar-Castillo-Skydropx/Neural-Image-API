import { build, Plugin, context } from "esbuild";
import fastGlob from "fast-glob";
import alias from "esbuild-plugin-alias";
import serve from "@es-exec/esbuild-plugin-serve";

const copyStaticFiles = require("esbuild-copy-static-files");
const handlebars = require("esbuild-plugin-handlebars");

const EsbuildPluginImportGlob = (): Plugin => ({
  name: "require-context",
  setup: (build) => {
    build.onResolve({ filter: /\*/ }, async (args) => {
      if (args.resolveDir === "") {
        return; // Ignore unresolvable paths
      }

      return {
        path: args.path,
        namespace: "import-glob",
        pluginData: {
          resolveDir: args.resolveDir,
        },
      };
    });

    build.onLoad({ filter: /.*/, namespace: "import-glob" }, async (args) => {
      const files = (
        await fastGlob(
          args.path.replace("@", require("path").resolve(__dirname, "src")),
          {
            cwd: args.pluginData.resolveDir,
          }
        )
      ).sort();

      let importerCode = `
        ${files
          .map(
            (module, index) =>
              `import * as module${index} from '${module.replace(".ts", "")}'`
          )
          .join(";")}

        const modules = [${files
          .map(
            (_, index) =>
              `Object.keys(module${index}).map(key => module${index}[key])`
          )
          .join(",")}].flat();

        export default modules;
      `;

      return { contents: importerCode, resolveDir: args.pluginData.resolveDir };
    });
  },
});

const isWatch = process.argv.includes("--watch");

(isWatch ? context : build)({
  external: [],
  entryPoints: ["./src/index.ts"],
  bundle: true,
  platform: "node",
  target: "ES2022",
  outfile: "./build/index.js",
  sourcemap: true,
  plugins: [
    EsbuildPluginImportGlob(),
    alias({
      "@": require("path").resolve(__dirname, "src"),
    }),
    handlebars(),
    ...(isWatch ? [serve({})] : []),
  ],
  tsconfig: "./tsconfig.json",
  format: "esm",
  logLevel: "info",
})
  .then((r: any) => {
    if (isWatch) r.watch();
  })
  .catch((e) => {
    console.error(e);
    process.exit(1);
  });
