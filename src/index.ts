import App from "@/app";

// Obtener el puerto del entorno o usar el predeterminado
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;

// Crear e iniciar la aplicación
const app = new App(PORT);
app.listen();

console.log(
  `Servidor de API de reconocimiento de imágenes iniciado en el puerto ${PORT}`
);
