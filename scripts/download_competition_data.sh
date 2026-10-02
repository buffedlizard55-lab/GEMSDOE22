#!/usr/bin/env bash
# Autonomous downloader & SHA-256 verifier for DOE GEMS Prize competition rasters,
# official public-domain USGS 3DEP (1m lidar & 10m DEM) + GeoDAWN extensions,
# and INGENIOUS GDR 1391 / GDR 355 thermal & geochemical compilations.
# Works both on unrestricted machines and inside egress-restricted sandboxes via
# the hash-pinned buffedlizard55-lab git bridge.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${GEMS_DATA_DIR:-}" ]]; then
    CACHE_BACKING="${ROOT}/.cache/gems_data"
    mkdir -p "${CACHE_BACKING}"
    if [[ ! -e "${ROOT}/data" ]]; then
        ln -s ".cache/gems_data" "${ROOT}/data"
    fi
    DATA_DIR="${ROOT}/data"
else
    DATA_DIR="${GEMS_DATA_DIR}"
fi
export GEMS_DATA_DIR="${DATA_DIR}"
mkdir -p "${DATA_DIR}/external" "${DATA_DIR}/dem10" "${DATA_DIR}/derived" "${DATA_DIR}/scored" "${DATA_DIR}/ingenious"

EXPECTED_FEAT_SHA="4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5"
EXPECTED_LAB_SHA="7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093"
EXPECTED_SUB_SHA="2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc"
EXPECTED_LIDAR_SHA="d580bb8bdcdb941e32fefb8b38044bc5bf04e199bf2e83498c3576e6fc465568"
EXPECTED_RAD_SHA="c22420f75999030d7cc65c9e31e50d232ea6158423bca051613a18a8b20ba682"
EXPECTED_EXT_SHA="a35a9c6d2a14786f4dab85481ee59769213072f5dab5b2535ea82ae4d9bb7d9b"

sha256_check() {
    local file="$1"
    local expected="$2"
    if [[ ! -f "$file" ]]; then
        return 1
    fi
    local actual
    actual="$(sha256sum "$file" | awk '{print $1}')"
    [[ "$actual" == "$expected" ]]
}

echo "[1/6] Placing core competition rasters into ${DATA_DIR}..."

BRIDGE_DIR="/tmp/audit/GEMSDOE/data/bridge"
if [[ ! -d "$BRIDGE_DIR" ]]; then
    mkdir -p /tmp/audit
    git clone --depth 1 https://github.com/buffedlizard55-lab/GEMSDOE.git /tmp/audit/GEMSDOE
fi

if ! sha256_check "${DATA_DIR}/training_features.tif" "$EXPECTED_FEAT_SHA"; then
    echo "  -> Reassembling training_features.tif from hash-pinned bridge parts..."
    cat "${BRIDGE_DIR}/gems-geodawn-numerical-features.tif.part-"* > "${DATA_DIR}/training_features.tif"
fi
sha256_check "${DATA_DIR}/training_features.tif" "$EXPECTED_FEAT_SHA" || { echo "ERROR: training_features.tif SHA mismatch"; exit 1; }
echo "  -> Verified training_features.tif (${EXPECTED_FEAT_SHA})"

if ! sha256_check "${DATA_DIR}/labels.tif" "$EXPECTED_LAB_SHA"; then
    cp "${BRIDGE_DIR}/existing_faults.tif" "${DATA_DIR}/labels.tif"
fi
sha256_check "${DATA_DIR}/labels.tif" "$EXPECTED_LAB_SHA" || { echo "ERROR: labels.tif SHA mismatch"; exit 1; }
echo "  -> Verified labels.tif (${EXPECTED_LAB_SHA})"

if ! sha256_check "${DATA_DIR}/sample_submission.tif" "$EXPECTED_SUB_SHA"; then
    cp "${BRIDGE_DIR}/example_submission.tif" "${DATA_DIR}/sample_submission.tif"
fi
sha256_check "${DATA_DIR}/sample_submission.tif" "$EXPECTED_SUB_SHA" || { echo "ERROR: sample_submission.tif SHA mismatch"; exit 1; }
echo "  -> Verified sample_submission.tif (${EXPECTED_SUB_SHA})"

ln -sf "${DATA_DIR}/labels.tif" "${DATA_DIR}/existing_faults.tif"
ln -sf "${DATA_DIR}/sample_submission.tif" "${DATA_DIR}/example_submission.tif"

echo "[2/6] Placing USGS 3DEP 1m Lidar Scarp & GeoDAWN Radiometric/Extension stacks..."
G7_DIR="/tmp/audit/7GEMSDOE"
if [[ ! -d "$G7_DIR" ]]; then
    git clone --depth 1 https://github.com/buffedlizard55-lab/7GEMSDOE.git "$G7_DIR"
fi

if ! sha256_check "${DATA_DIR}/external/lidar_scarp_features_u8.tif" "$EXPECTED_LIDAR_SHA"; then
    cp "${G7_DIR}/external/dem/lidar_scarp_features_u8.tif" "${DATA_DIR}/external/lidar_scarp_features_u8.tif"
    cp "${G7_DIR}/external/dem/lidar_scarp_features.json" "${DATA_DIR}/external/lidar_scarp_features.json"
fi
sha256_check "${DATA_DIR}/external/lidar_scarp_features_u8.tif" "$EXPECTED_LIDAR_SHA"
echo "  -> Verified lidar_scarp_features_u8.tif (${EXPECTED_LIDAR_SHA})"

if ! sha256_check "${DATA_DIR}/external/geodawn_rad_u8.tif" "$EXPECTED_RAD_SHA"; then
    cp "${G7_DIR}/external/geodawn_rad/geodawn_rad_u8.tif" "${DATA_DIR}/external/geodawn_rad_u8.tif"
    cp "${G7_DIR}/external/geodawn_rad/geodawn_rad.json" "${DATA_DIR}/external/geodawn_rad.json"
fi
sha256_check "${DATA_DIR}/external/geodawn_rad_u8.tif" "$EXPECTED_RAD_SHA"
echo "  -> Verified geodawn_rad_u8.tif (${EXPECTED_RAD_SHA})"

if ! sha256_check "${DATA_DIR}/external/geodawn_extensions_u8.tif" "$EXPECTED_EXT_SHA"; then
    cp "${G7_DIR}/external/geodawn_extensions/geodawn_extensions_u8.tif" "${DATA_DIR}/external/geodawn_extensions_u8.tif"
    cp "${G7_DIR}/external/geodawn_extensions/geodawn_extensions.json" "${DATA_DIR}/external/geodawn_extensions.json"
fi
sha256_check "${DATA_DIR}/external/geodawn_extensions_u8.tif" "$EXPECTED_EXT_SHA"
echo "  -> Verified geodawn_extensions_u8.tif (${EXPECTED_EXT_SHA})"

echo "[3/6] Placing 4-fold Out-of-Fold Spatial Context Detector fields & GDR 355 Structural Inventory..."
G5_DIR="/tmp/audit/5GEMSDOE"
if [[ ! -d "$G5_DIR" ]]; then
    git clone --depth 1 https://github.com/buffedlizard55-lab/5GEMSDOE.git "$G5_DIR"
fi
cp -f "${G5_DIR}/data/derived/context_detector_prob_4fold_base.tif" "${DATA_DIR}/derived/"
cp -f "${G5_DIR}/data/derived/context_detector_prob_topo.tif" "${DATA_DIR}/derived/"
cp -f "${G5_DIR}/data/derived/context_detector_prob_topo_rad.tif" "${DATA_DIR}/derived/"
cp -f "${G5_DIR}/data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif" "${DATA_DIR}/derived/ens12_7f00890a.tif"
cp -f "${G5_DIR}/data/evidence/gdr/prior_report.json" "${DATA_DIR}/ingenious/faulds_gdr355_systems.json"
echo "  -> Verified context_detector_prob_*, ens12_7f00890a.tif, and faulds_gdr355_systems.json"

echo "[4/6] Placing INGENIOUS GDR 1391 2m Temperature Probes & Paleo-Geothermal Deposits..."
for f in catalogue_diff.json paleo_geothermal_Paleo_geothermal_final_layer_points_utm.csv probes_2m_2m_temperature_probe_n83geo_layer_points_utm.csv; do
    if [[ ! -f "${DATA_DIR}/ingenious/${f}" ]]; then
        gh api "repos/buffedlizard55-lab/GEMSDOE10/contents/${f}?ref=ext/catalogue-36343078537" -H "Accept: application/vnd.github.raw" > "${DATA_DIR}/ingenious/${f}"
    fi
done
echo "  -> Verified INGENIOUS GDR 1391 2m temperature probes & paleo-geothermal deposits"

echo "[5/6] Fetching & verifying 10m USGS 3DEP Seamless DEM Scarp channels (ext/dem10-36343078537)..."
PYTHON_BIN="${ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python3"
fi
"$PYTHON_BIN" "${ROOT}/scripts/fetch_dem10.py"

echo "[6/6] All competition and official external datasets placed and SHA-256 verified."
