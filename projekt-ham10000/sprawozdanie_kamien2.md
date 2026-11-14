# Klasyfikacja zmian skórnych z wykorzystaniem CNN, ze szczególnym uwzględnieniem wykrywania czerniaka

*Sprawozdanie – kamień milowy 2 · Michał Kowalik · listopad 2026*

## 1. Cel etapu

Celem etapu była implementacja i przeprowadzenie eksperymentów E1–E8 zaplanowanych w kamieniu 1, weryfikacja hipotez H1–H4 oraz wybór najlepszego wariantu modelu. Wszystkie wyniki podane są na zbiorze walidacyjnym. Zbiór testowy nie był używany i posłuży do oceny końcowej w kamieniu 3.

## 2. Implementacja

| Element | Rozwiązanie |
|---|---|
| Dane | podział z kamienia 1 (`splits.csv`, po `lesion_id`); obrazy 256 × 256, wejście sieci 224 × 224 |
| Modele | prosta CNN (4 bloki conv–BN–ReLU–pool, 0,4 mln parametrów), ResNet50 (23,5 mln), EfficientNet-B0 (4,0 mln); sieci wstępnie wytrenowane na ImageNet z ostatnią warstwą wymienioną na 7 klas |
| Augmentacja | brak / podstawowa (losowe kadrowanie, odbicia, obrót ±20°, jasność i kontrast) / RandAugment |
| Niezbalansowanie | zwykła CE / ważona CE (waga klasy = liczba zdjęć / (7 · liczność klasy)) / oversampling (`WeightedRandomSampler`) |
| Trening | AdamW (lr 1e-4, dla CNN od zera 1e-3), harmonogram cosinusowy, 15 epok, batch 32, mixed precision |
| Wybór epoki | zapis modelu z najwyższym balanced accuracy na walidacji |
| Powtórzenia | 3 ziarna losowe na wariant, łącznie 24 treningi |
| Sprzęt | RTX 3060 Ti; 4–7 min na trening, łącznie 2 h 10 min |

Kod znajduje się w notebooku `kamien2.ipynb`.

## 3. Wyniki

| # | Wariant | Balanced acc. | Macro-F1 | Recall mel | AUC |
|---|---|---|---|---|---|
| E1 | prosta CNN (baseline) | 0,526 ± 0,003 | 0,396 | 0,584 | 0,866 |
| E2 | ResNet50 | 0,783 ± 0,005 | 0,729 | 0,608 | 0,967 |
| E3 | EfficientNet-B0 (referencja) | 0,779 ± 0,002 | 0,704 | 0,671 | 0,960 |
| E4 | bez augmentacji | 0,775 ± 0,006 | 0,716 | 0,649 | 0,954 |
| E5 | **RandAugment** | **0,797 ± 0,013** | **0,739** | **0,691** | **0,967** |
| E6 | zwykła CE | 0,714 ± 0,002 | 0,731 | 0,526 | 0,969 |
| E7 | oversampling | 0,780 ± 0,004 | 0,732 | 0,699 | 0,963 |
| E8 | podział losowy (wyciek) | 0,856 ± 0,005 | 0,771 | 0,770 | 0,972 |

Punkt odniesienia – model zawsze wskazujący nv – osiąga balanced accuracy 0,143.

![Wyniki eksperymentów](figures/k2_wyniki.png)

![Krzywe uczenia](figures/k2_krzywe.png)

## 4. Weryfikacja hipotez

**H1 – transfer learning > CNN od zera: potwierdzona.** Sieci wstępnie wytrenowane osiągają ok. 25 punktów procentowych wyższe balanced accuracy (0,526 → 0,78). ResNet50 i EfficientNet-B0 dają praktycznie ten sam wynik (różnica mniejsza niż rozrzut między seedami), ale EfficientNet ma 6 razy mniej parametrów, trenuje się szybciej i lepiej wykrywa czerniaka (0,67 vs 0,61). Krzywa E1 rosła do ostatniej epoki, więc baseline był lekko niedotrenowany – przy tej różnicy nie zmienia to jednak wniosku.

**H2 – obsługa niezbalansowania zwiększa recall czerniaka: potwierdzona.** Recall czerniaka rośnie z 0,526 (zwykła CE) do 0,671 (ważona CE) i 0,699 (oversampling). Model ze zwykłą CE ma przy tym najwyższe AUC (0,969), ale wyraźnie faworyzuje nv (recall nv 0,95 wobec 0,80 w E3). Oznacza to, że ważenie i oversampling nie poprawiają samej zdolności rozróżniania klas, tylko przesuwają granicę decyzji w stronę klas rzadkich. Obie metody dają zbliżone wyniki.

**H3 – augmentacja poprawia wyniki: potwierdzona częściowo.** RandAugment dał najlepszy wynik (0,797), wyższy od E3 w każdym z trzech seedów. Podstawowa augmentacja w najlepszej epoce wypadła podobnie jak brak augmentacji (0,779 vs 0,775), jednak krzywe uczenia pokazują, że bez augmentacji model zaczyna się przeuczać już po 3.–5. epoce. Szczególnej poprawy dla klas rzadkich nie zaobserwowano – zbiór walidacyjny zawiera zbyt mało ich zdjęć (np. 18 zdjęć df), by różnice per klasa były wiarygodne.

**H4 – losowy podział zawyża wyniki: potwierdzona.** Przy losowym podziale balanced accuracy wzrasta o 7,7 punktu procentowego (0,856 vs 0,779), a recall czerniaka o 10 punktów (0,770 vs 0,671). Recall klas df i vasc dochodzi do 0,98, co zgadza się z największym wyciekiem zmierzonym dla tych klas w kamieniu 1. Wynik potwierdza konieczność podziału po `lesion_id`.

## 5. Wybór modelu

Wybrano wariant **E5: EfficientNet-B0 + RandAugment + ważona CE** – najwyższe średnie balanced accuracy spośród wariantów z poprawnym podziałem (E8 pominięto) przy jednocześnie wysokim recallu czerniaka. Do kamienia 3 używany jest checkpoint `models/E5_s0.pt` (balanced accuracy 0,815, recall czerniaka 0,705). Jest to najlepszy z trzech seedów, więc wynik na walidacji jest lekko optymistyczny – rzetelną ocenę da zbiór testowy.

## 6. Ograniczenia i dalsze kroki

- Najlepsze epoki E1 i E5 wypadały pod koniec treningu, więc dłuższy trening mógłby nieco poprawić te warianty.
- Metryki dla klas rzadkich są niestabilne ze względu na małą liczbę zdjęć w walidacji (df: 18, vasc: 21).
- Recall czerniaka ok. 0,70 oznacza, że model pomija ok. 3 na 10 czerniaków.

**Kamień 3:** ocena modelu E5 na zbiorze testowym, macierz pomyłek, analiza Grad-CAM z wykorzystaniem masek segmentacji (H5), analiza progu decyzyjnego dla czerniaka oraz aplikacja Gradio.
