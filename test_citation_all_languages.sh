#!/bin/bash

# Test citation preservation for all supported languages
# Tests that [0001] style citations are preserved correctly

set -euo pipefail

FILE="/home/green/Downloads/citation (1).docx"
TRANSLATION_API="http://localhost:8003"

# All supported languages
declare -A LANGUAGES=(
    ["chinese (mandarin)"]="zh"
    ["simplified chinese"]="zh"
    ["german"]="de"
    ["french"]="fr"
    ["spanish"]="es"
    ["italian"]="it"
    ["english"]="en"
    ["japanese"]="ja"
    ["korean"]="ko"
    ["swedish"]="sv"
    ["danish"]="da"
    ["finnish"]="fi"
    ["dutch"]="nl"
)

# Test results
PASSED=0
FAILED=0
FAILED_LANGUAGES=()

echo "=========================================="
echo "Citation Preservation Test - All Languages"
echo "=========================================="
echo "Testing file: $FILE"
echo "Total languages to test: ${#LANGUAGES[@]}"
echo ""

# Test each language
for lang_name in "${!LANGUAGES[@]}"; do
    echo "----------------------------------------"
    echo "Testing: $lang_name (${LANGUAGES[$lang_name]})"
    echo "----------------------------------------"
    
    # Submit translation job
    RESP=$(curl -sS -X POST \
        -F "files=@${FILE};type=application/vnd.openxmlformats-officedocument.wordprocessingml.document" \
        -F "prompt=Translate to ${lang_name}" \
        -F "target_language=${lang_name}" \
        -F "enable_evaluation=false" \
        "${TRANSLATION_API}/translate_file_convo")
    
    # Check if job was created
    JOB=$(echo "$RESP" | jq -r ".jobs[0].job_id" 2>/dev/null || true)
    
    if [ -z "$JOB" ]; then
        echo "❌ FAILED: No job_id returned"
        FAILED=$((FAILED + 1))
        FAILED_LANGUAGES+=("$lang_name")
        continue
    fi
    
    echo "  Job ID: $JOB"
    echo "  Waiting for completion..."
    
    # Wait for job completion (max 120 seconds)
    COMPLETED=false
    for i in $(seq 1 120); do
        ST=$(curl -sS "${TRANSLATION_API}/status/$JOB" | jq -r ".status" 2>/dev/null || echo "UNKNOWN")
        
        if [ "$ST" = "COMPLETED" ]; then
            COMPLETED=true
            break
        elif [ "$ST" = "FAILED" ]; then
            echo "  ❌ Job failed"
            FAILED=$((FAILED + 1))
            FAILED_LANGUAGES+=("$lang_name")
            continue 2
        fi
        
        sleep 1
    done
    
    if [ "$COMPLETED" = "false" ]; then
        echo "  ❌ TIMEOUT: Job did not complete in 120 seconds"
        FAILED=$((FAILED + 1))
        FAILED_LANGUAGES+=("$lang_name")
        continue
    fi
    
    # Get preview text
    RES=$(curl -sS "${TRANSLATION_API}/status/$JOB")
    DL=$(echo "$RES" | jq -r ".download_id" 2>/dev/null || true)
    
    if [ -z "$DL" ]; then
        echo "  ❌ FAILED: No download_id"
        FAILED=$((FAILED + 1))
        FAILED_LANGUAGES+=("$lang_name")
        continue
    fi
    
    PREV=$(curl -sS "${TRANSLATION_API}/preview/$DL")
    PREVIEW_TEXT=$(echo "$PREV" | jq -r ".preview" 2>/dev/null || echo "")
    
    # Check if [0001] citation is preserved
    if echo "$PREVIEW_TEXT" | grep -q "\[0001\]"; then
        echo "  ✅ PASS: [0001] citation preserved correctly"
        PASSED=$((PASSED + 1))
    elif echo "$PREVIEW_TEXT" | grep -q "\[0\]"; then
        echo "  ⚠️  WARNING: [0001] was changed to [0] - leading zeros lost"
        FAILED=$((FAILED + 1))
        FAILED_LANGUAGES+=("$lang_name")
    elif echo "$PREVIEW_TEXT" | grep -q "\[.*1.*\]"; then
        echo "  ⚠️  PARTIAL: Citation found but format may be incorrect"
        # Check what the actual format is
        CITATION_FORMAT=$(echo "$PREVIEW_TEXT" | grep -o "\[[^]]*\]" | head -1 || echo "none")
        echo "  Found citation format: $CITATION_FORMAT"
        FAILED=$((FAILED + 1))
        FAILED_LANGUAGES+=("$lang_name")
    else
        echo "  ❌ FAILED: No citation found in preview"
        echo "  Preview snippet (first 200 chars): ${PREVIEW_TEXT:0:200}"
        FAILED=$((FAILED + 1))
        FAILED_LANGUAGES+=("$lang_name")
    fi
    
    echo ""
done

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Total languages tested: ${#LANGUAGES[@]}"
echo "✅ Passed: $PASSED"
echo "❌ Failed: $FAILED"
echo ""

if [ $FAILED -gt 0 ]; then
    echo "Failed languages:"
    for lang in "${FAILED_LANGUAGES[@]}"; do
        echo "  - $lang"
    done
    echo ""
    exit 1
else
    echo "🎉 All languages passed citation preservation test!"
    exit 0
fi


