# Entrenamiento largo del dron de entrenamiento (Holybro X650, física de 4 motores) con CMA-ES (eval/evolve.py
# --opt cmaes, el optimizador por defecto), independiente de la sesión: se puede cerrar todo y sigue. Solo los pilotos
# que se entrenan: clásico e híbrido ("fábrica" es la referencia; el reactivo puro se retiró).
#   Ronda 1: λ = 14 individuos por generación, 30 generaciones, 3 mundos por escenario (18 carreras por individuo).
#   Ronda 2: partiendo de lo adoptado, λ = 16, 40 generaciones y 4 mundos por escenario (menos sobreajuste).
# Cada entrenamiento se valida en semillas nuevas y solo se adopta si mejora sin chocar más y dentro de los límites
# realistas (eval/apply_evolved.py). Progreso en eval/entrenamiento.log. Para pararlo: Get-Process python | Stop-Process
# (16 procesos de los 20 núcleos: con 18 y bancos de pruebas a la vez, el grupo de procesos llegó a morir)
$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONIOENCODING = "utf-8"
$log = "eval\entrenamiento.log"
function Log($msg) { $msg | Out-File -Encoding utf8 -Append $log }
Log "=== X650 con CMA-ES: ronda 1: $(Get-Date -Format 'yyyy-MM-dd HH:mm') ==="
cmd /c "python -u eval/train_all.py --pop 14 --gens 30 --seeds 3 --jobs 16 >> eval\entrenamiento.log 2>&1"
Log "=== X650 con CMA-ES: ronda 2: $(Get-Date -Format 'yyyy-MM-dd HH:mm') ==="
cmd /c "python -u eval/train_all.py --pop 16 --gens 40 --seeds 4 --jobs 16 >> eval\entrenamiento.log 2>&1"
Log "=== fin: $(Get-Date -Format 'yyyy-MM-dd HH:mm') ==="
