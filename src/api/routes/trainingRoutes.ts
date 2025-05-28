import { Router } from "express";
import { upload, handleMulterError } from "@/api/middlewares/imageUpload";
import { TrainingController } from "@/api/controllers/TrainingController";

const router = Router();
const trainingController = new TrainingController();

/**
 * @route POST /api/training/train
 * @desc Entrena la red neuronal con imágenes etiquetadas
 * @access Public
 */
router.post(
  "/train",
  upload.array("images", 50),
  handleMulterError,
  trainingController.trainNetwork
);

/**
 * @route GET /api/training/status
 * @desc Obtiene el estado del entrenamiento
 * @access Public
 */
router.get("/status", trainingController.getTrainingStatus);

/**
 * @route POST /api/training/save
 * @desc Guarda el modelo entrenado
 * @access Public
 */
router.post("/save", trainingController.saveModel);

/**
 * @route POST /api/training/load
 * @desc Carga un modelo previamente entrenado
 * @access Public
 */
router.post("/load", trainingController.loadModel);

export const trainingRoutes = router;
