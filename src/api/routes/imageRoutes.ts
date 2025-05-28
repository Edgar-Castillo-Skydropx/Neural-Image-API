import { Router } from "express";
import {
  upload,
  handleMulterError,
  validateImageUpload,
} from "@/api/middlewares/imageUpload";
import { ImageController } from "@/api/controllers/ImageController";

const router = Router();
const imageController = new ImageController();

/**
 * @route POST /api/images/classify
 * @desc Clasifica una imagen subida
 * @access Public
 */
router.post(
  "/classify",
  upload.single("image"),
  handleMulterError,
  validateImageUpload,
  imageController.classifyImage
);

/**
 * @route GET /api/images/status
 * @desc Verifica el estado del servicio de clasificación
 * @access Public
 */
router.get("/status", imageController.getStatus);

export const imageRoutes = router;
