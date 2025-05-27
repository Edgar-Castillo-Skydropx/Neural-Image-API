import { Request, Response, NextFunction } from 'express';

/**
 * Middleware para manejo centralizado de errores
 */
export const errorHandler = (
  error: Error,
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  console.error(`Error: ${error.message}`);
  console.error(error.stack);

  // Determinar el código de estado HTTP
  let statusCode = 500;
  let errorMessage = 'Error interno del servidor';

  // Personalizar respuesta según el tipo de error
  if (error.name === 'ValidationError') {
    statusCode = 400;
    errorMessage = error.message;
  } else if (error.name === 'UnauthorizedError') {
    statusCode = 401;
    errorMessage = 'No autorizado';
  } else if (error.name === 'ForbiddenError') {
    statusCode = 403;
    errorMessage = 'Acceso prohibido';
  } else if (error.name === 'NotFoundError') {
    statusCode = 404;
    errorMessage = 'Recurso no encontrado';
  }

  // Enviar respuesta de error
  res.status(statusCode).json({
    success: false,
    error: {
      message: errorMessage,
      code: statusCode
    }
  });
};
