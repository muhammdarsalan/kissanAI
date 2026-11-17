## Results so far
- Model: MobileNetV3-Large (timm, ImageNet-pretrained), 224 px, 17 classes (tomato, potato, maize), trained on PlantVillage with a leaf-grouped split, class-weighted loss, two-phase training (head 3 epochs, fine-tune 7).
- Lab validation (PlantVillage): accuracy 97.8%, macro-F1 0.977.
- Field test (PlantDoc test split, 111 images of 13 classes, CC BY 4.0): accuracy 27.0% (about 20-36% with n=111).
- Confidence thresholds do not help on field images: accuracy on kept predictions stays about 28% from 0.5 to 0.95.
- Unknown crops (125 PlantDoc images of other crops): only 21% rejected at 0.5, 82% at 0.9.
- Main errors: crop mix-ups (potato vs tomato), early vs late blight, corn lesion types, watermarks and busy scenes.
- Limitations: corn leaf groups use a proxy (leaf map covers no corn); PlantDoc test is tiny and noisy; 4 classes have no field images.
## Next
1. Fine-tune with PlantDoc train images (deduplicated, 20% held out as field-val).
2. Add FieldPlant as a second untouched field test.
3. Re-choose confidence thresholds on field-val.
4. Compare a second backbone.
5. Wheat and more KPK crops, then RF models, grounded advisory chat, Streamlit app, evaluation.

## Field fine-tuning (PlantDoc train, 1,219 images kept, 14 near-duplicates of test dropped, x5 repeat)
- Lab val: 97.9% accuracy, macro-F1 0.978 (unchanged).
- Field test (PlantDoc test, 111 images): 27.0% -> 44.1% (about 35-53%).
- Confidence now informative: 65.1% accuracy on kept predictions at 0.95 (39% coverage), vs. flat 28% before.
- Unknown-crop rejection got worse (63% at 0.9, was 82%): needs an "other" class or an OOD score.
- Remaining weak points: Northern leaf blight vs Gray leaf spot, potato early vs late blight, mosaic virus (10%).
- PlantDoc test used twice; later development should use FieldPlant and field-val only.


## Field fine-tune: statistical check
- Paired comparison on 111 PlantDoc test images: 28 images fixed, 9 broken by fine-tuning (McNemar exact p = 0.0026). Accuracy 27.0% -> 44.1% (+17.1 points, 95% bootstrap CI +7.2 to +27.0; after-accuracy CI 35.1-53.2%).
- Crop identified correctly by the model alone: 92.8% (was 82%). Cross-crop errors: 8 of 111 (was 20).
- Accuracy within the right crop is 47.6%, so the main remaining problem is disease confusion within a crop, not crop confusion.

## Crop-conditioned field evaluation (PlantDoc test, 111 images)
- Crop selected by farmer (predictions restricted to that crop): top-1 45.9% (about 37-55%), correct disease in top 3: 88.3% (about 81-93%).
- Accuracy on kept predictions with crop selected: 62% at confidence 0.90 (55% of images kept) and 0.95 (45% kept).
- Design decision: the app shows the top 3 candidates for the selected crop plus an expert-confirmation note; it does not present a single confident diagnosis.