import multer from "multer";
import path from "path";
import { Request, Response, NextFunction } from "express";

// Configuración de almacenamiento para multer
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, path.join(__dirname, "../uploads"));
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    const ext = path.extname(file.originalname);
    cb(null, file.fieldname + "-" + uniqueSuffix + ext);
  },
});

// Filtro para validar tipos de archivos
const fileFilter = (
  req: Request,
  file: Express.Multer.File,
  cb: multer.FileFilterCallback
) => {
  // Aceptar solo imágenes
  if (file.mimetype.startsWith("image/")) {
    cb(null, true);
  } else {
    cb(new Error("El archivo debe ser una imagen válida"));
  }
};

// Configuración de límites
const limits = {
  fileSize: 5 * 1024 * 1024, // 5MB máximo
};

// Crear instancia de multer
export const upload = multer({
  storage,
  fileFilter,
  limits,
});

/**
 * Middleware para manejar errores de multer
 */
export const handleMulterError = (
  err: Error,
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  if (err instanceof multer.MulterError) {
    let message = "Error al subir archivo";

    if (err.code === "LIMIT_FILE_SIZE") {
      message = "El archivo excede el tamaño máximo permitido (5MB)";
    }

    res.status(400).json({
      success: false,
      error: {
        message,
        code: 400,
        err,
      },
    });
  } else if (err) {
    next(err);
  } else {
    next();
  }
};

/**
 * Middleware para validar que se haya subido una imagen
 */
export const validateImageUpload = (
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  if (!req.file) {
    res.status(400).json({
      success: false,
      error: {
        message: "No se ha proporcionado ninguna imagen",
        code: 400,
      },
    });
    return;
  }

  next();
};
