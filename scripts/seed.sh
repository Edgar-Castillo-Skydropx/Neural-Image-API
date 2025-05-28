#!/bin/bash

# Script para ejecutar el poblador de base de datos
echo "Ejecutando script de poblado de base de datos..."

# Verificar que Node.js y npm estén instalados
if ! command -v node &> /dev/null || ! command -v npm &> /dev/null; then
    echo "Error: Node.js y npm son requeridos para ejecutar este script."
    exit 1
fi

# Verificar que TypeScript esté instalado
if ! command -v tsc &> /dev/null; then
    echo "Instalando TypeScript..."
    npm install -g typescript
fi

# Verificar que ts-node esté instalado
if ! command -v ts-node &> /dev/null; then
    echo "Instalando ts-node..."
    npm install -g ts-node
fi

# Verificar que estamos en el directorio correcto
if [ ! -f "package.json" ]; then
    echo "Error: Este script debe ejecutarse desde el directorio raíz del proyecto."
    exit 1
fi

# Verificar que el archivo .env existe
if [ ! -f ".env" ]; then
    echo "Advertencia: No se encontró el archivo .env. Copiando desde .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
    else
        echo "Error: No se encontró .env.example. Por favor, crea un archivo .env manualmente."
        exit 1
    fi
fi

# Ejecutar el script de poblado
echo "Poblando la base de datos..."
ts-node scripts/seed-database.ts

# Verificar si la ejecución fue exitosa
if [ $? -eq 0 ]; then
    echo "Base de datos poblada exitosamente."
else
    echo "Error al poblar la base de datos. Revisa los logs para más detalles."
    exit 1
fi

echo "Proceso completado."
