import os
import re
import json
import pandas as pd
import numpy as np
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

def build_knowledge_base(dir_path, output_path="circuit_knowledge_base"):
    """
    Создает векторную базу знаний из технической документации по электронным схемам.
    
    Args:
        dir_path: Путь к директории с документами
        output_path: Путь для сохранения векторной базы
    
    Returns:
        Статус операции и статистику обработанных документов
    """
    documents = []
    stats = {"pdf": 0, "txt": 0, "csv": 0, "skipped": 0}
    if not os.path.exists(dir_path):
        return {"status": "error", "message": f"Директория {dir_path} не существует"}
    for root, _, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.lower().endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                    stats["pdf"] += 1
                elif file.lower().endswith(".txt"):
                    loader = TextLoader(file_path)
                    documents.extend(loader.load())
                    stats["txt"] += 1
                elif file.lower().endswith(".csv"):
                    loader = CSVLoader(file_path)
                    documents.extend(loader.load())
                    stats["csv"] += 1
            except Exception as e:
                stats["skipped"] += 1
                print(f"Ошибка при обработке файла {file_path}: {str(e)}")
    
    if not documents:
        return {"status": "error", "message": "Не найдено документов для обработки"}
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(output_path)
        return {
            "status": "success", 
            "message": f"База знаний создана и сохранена в {output_path}", 
            "stats": {
                "processed_files": stats,
                "total_chunks": len(chunks),
                "total_documents": len(documents)
            }
        }
    except Exception as e:
        return {"status": "error", "message": f"Ошибка при создании базы знаний: {str(e)}"}

def search_knowledge_base(query, kb_path="circuit_knowledge_base", top_k=5):
    """
    Ищет информацию в векторной базе знаний.
    
    Args:
        query: Поисковый запрос
        kb_path: Путь к векторной базе знаний
        top_k: Количество результатов для возврата
    
    Returns:
        Список релевантных документов
    """
    try:
        if not os.path.exists(kb_path):
            return {"status": "error", "message": f"База знаний в {kb_path} не найдена"}
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        db = FAISS.load_local(kb_path, embeddings)
        docs = db.similarity_search(query, k=top_k)
        results = []
        for i, doc in enumerate(docs):
            result = {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", None)
            }
            results.append(result)
        
        return {
            "status": "success",
            "results": results
        }
    except Exception as e:
        return {"status": "error", "message": f"Ошибка при поиске: {str(e)}"}

def add_document_to_kb(document_path, kb_path="circuit_knowledge_base"):
    """
    Добавляет новый документ в существующую базу знаний.
    
    Args:
        document_path: Путь к документу для добавления
        kb_path: Путь к векторной базе знаний
    
    Returns:
        Статус операции
    """
    try:
        if not os.path.exists(kb_path):
            return {"status": "error", "message": f"База знаний в {kb_path} не найдена"}
        documents = []
        if document_path.lower().endswith(".pdf"):
            loader = PyPDFLoader(document_path)
            documents = loader.load()
        elif document_path.lower().endswith(".txt"):
            loader = TextLoader(document_path)
            documents = loader.load()
        elif document_path.lower().endswith(".csv"):
            loader = CSVLoader(document_path)
            documents = loader.load()
        else:
            return {"status": "error", "message": "Неподдерживаемый формат файла"}
        
        if not documents:
            return {"status": "error", "message": "Не удалось загрузить документ"}
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        db = FAISS.load_local(kb_path, embeddings)
        db.add_documents(chunks)
        db.save_local(kb_path)
        
        return {
            "status": "success", 
            "message": f"Документ {document_path} добавлен в базу знаний",
            "chunks_added": len(chunks)
        }
    except Exception as e:
        return {"status": "error", "message": f"Ошибка при добавлении документа: {str(e)}"}

def create_sample_circuit_knowledge():
    """
    Создает пример базы знаний с основной информацией об электронных схемах.
    
    Returns:
        Статус операции
    """
    try:
        os.makedirs("temp_docs", exist_ok=True)
        stability_info = """
        # Анализ стабильности электронных схем
        
        Стабильность электронной схемы - ключевой параметр, определяющий надежность работы устройства.
        
        ## Основные критерии стабильности
        
        1. **Запас по фазе** - должен быть не менее 45° для хорошей стабильности, предпочтительно 60° и выше.
           - Менее 45°: недостаточная стабильность
           - 45°-60°: приемлемая стабильность
           - Более 60°: хорошая стабильность
        
        2. **Запас по усилению** - рекомендуется не менее 10 дБ.
           - Менее 6 дБ: низкий запас
           - 6-10 дБ: минимально допустимый запас
           - Более 10 дБ: хороший запас
        
        ## Типичные проблемы со стабильностью
        
        1. Недостаточное демпфирование в LC-контурах
        2. Неправильная компенсация в операционных усилителях
        3. Паразитные обратные связи
        4. Резонансные явления
        
        ## Методы повышения стабильности
        
        1. Добавление последовательных резисторов в цепи обратной связи
        2. Использование компенсирующих конденсаторов
        3. Разделение земляных цепей аналоговой и цифровой части
        4. Использование метода доминирующего полюса
        """
        
        with open("temp_docs/stability_guide.txt", "w", encoding="utf-8") as f:
            f.write(stability_info)
        noise_info = """
        # Шумовой анализ электронных схем
        
        Шумовые характеристики определяют минимальный уровень сигнала, который может быть обработан схемой.
        
        ## Основные типы шумов
        
        1. **Тепловой шум** - возникает из-за хаотического движения электронов при ненулевой температуре.
           - Формула: Vn = sqrt(4kTRB), где:
             - k - постоянная Больцмана
             - T - абсолютная температура
             - R - сопротивление
             - B - полоса пропускания
        
        2. **Дробовой шум** - возникает из-за дискретной природы электрического тока.
           - Формула: In = sqrt(2qIB), где:
             - q - заряд электрона
             - I - ток
             - B - полоса пропускания
        
        3. **1/f шум (фликкер-шум)** - шум, спектральная плотность которого обратно пропорциональна частоте.
        
        ## Параметры шумового анализа
        
        1. **Коэффициент шума (NF)** - отношение SNR на входе к SNR на выходе.
           - NF = 1 (0 дБ): идеальная схема без добавления шума
           - NF = 2 (3 дБ): схема добавляет столько же шума, сколько было на входе
        
        2. **Эквивалентная шумовая температура** - характеризует шумовые свойства в температурных единицах.
        
        3. **Спектральная плотность шума** - шум в единичной полосе частот.
        
        ## Типичные значения коэффициента шума
        
        1. Пассивные компоненты (резисторы, конденсаторы): близко к 0 дБ
        2. Малошумящие предусилители: 0.5-2 дБ
        3. Стандартные операционные усилители: 3-10 дБ
        4. Активные смесители: 7-15 дБ
        
        ## Методы снижения шума
        
        1. Использование малошумящих компонентов
        2. Оптимизация импедансов
        3. Экранирование
        4. Фильтрация по питанию
        5. Уменьшение полосы пропускания до необходимого минимума
        """
        
        with open("temp_docs/noise_analysis.txt", "w", encoding="utf-8") as f:
            f.write(noise_info)
        
        transient_info = """
        # Анализ переходных процессов
        
        Переходные процессы характеризуют динамическое поведение схемы при изменении входных сигналов.
        
        ## Основные параметры переходных процессов
        
        1. **Время нарастания (Rise Time)** - время, за которое сигнал возрастает с 10% до 90% от установившегося значения.
           - Типичные значения:
             - Высокоскоростные схемы: <1 нс
             - Стандартные цифровые схемы: 1-10 нс
             - Аналоговые схемы: 0.1-100 мкс
        
        2. **Время спада (Fall Time)** - время, за которое сигнал спадает с 90% до 10% от установившегося значения.
        
        3. **Перерегулирование (Overshoot)** - превышение сигналом установившегося значения, выраженное в процентах.
           - Допустимые значения:
             - Критичные схемы: <5%
             - Стандартные схемы: 5-15%
             - Некритичные схемы: до 30%
        
        4. **Время установления (Settling Time)** - время, необходимое для того, чтобы сигнал вошел и остался в заданной полосе около установившегося значения (обычно ±5% или ±1%).
        
        5. **Задержка распространения (Propagation Delay)** - время между изменением входного сигнала и соответствующим изменением выходного сигнала.
        
        ## Факторы, влияющие на переходные процессы
        
        1. **Постоянные времени цепей** - RC, RL или LC постоянные времени определяют скорость реакции цепи.
        
        2. **Полоса пропускания усилителя** - связана с временем нарастания соотношением Tr ≈ 0.35/BW.
        
        3. **Скорость нарастания (Slew Rate)** - максимальная скорость изменения выходного напряжения усилителя.
        
        4. **Входная и выходная емкость** - паразитные емкости замедляют переходные процессы.
        
        ## Типичные проблемы с переходными процессами
        
        1. Недостаточное демпфирование, приводящее к колебаниям
        2. Ограничение скорости нарастания в операционных усилителях
        3. Искажение формы сигнала из-за нелинейностей
        4. Звон (ringing) на фронтах импульсов
        
        ## Способы улучшения переходных характеристик
        
        1. Оптимизация демпфирования (обычно коэффициент демпфирования 0.7 даёт оптимальный отклик)
        2. Использование быстродействующих компонентов
        3. Минимизация паразитных ёмкостей и индуктивностей
        4. Согласование импедансов для минимизации отражений
        """
        
        with open("temp_docs/transient_analysis.txt", "w", encoding="utf-8") as f:
            f.write(transient_info)
        frequency_info = """
        # Частотный анализ (AC-анализ)
        
        Частотный анализ позволяет оценить поведение схемы в частотной области.
        
        ## Основные характеристики
        
        1. **Амплитудно-частотная характеристика (АЧХ)** - зависимость амплитуды выходного сигнала от частоты входного.
        
        2. **Фазо-частотная характеристика (ФЧХ)** - зависимость фазового сдвига между выходным и входным сигналами от частоты.
        
        3. **Полоса пропускания** - диапазон частот, в котором АЧХ не опускается ниже уровня -3 дБ от максимального.
           - Аудиосхемы: 20 Гц - 20 кГц
           - ВЧ-схемы: до сотен МГц или ГГц
           - Широкополосные усилители: от постоянного тока до десятков МГц
        
        4. **Частота единичного усиления** - частота, на которой коэффициент усиления равен 1 (0 дБ).
        
        5. **Частота среза** - частота, на которой усиление падает на 3 дБ от максимального.
        
        ## Типы частотных характеристик
        
        1. **Фильтр нижних частот (ФНЧ)** - пропускает частоты ниже частоты среза.
           - Скорость спада: 6 дБ/октаву для фильтра 1-го порядка, 12 дБ/октаву для 2-го порядка и т.д.
        
        2. **Фильтр верхних частот (ФВЧ)** - пропускает частоты выше частоты среза.
        
        3. **Полосовой фильтр** - пропускает частоты в определенной полосе.
           - Характеризуется центральной частотой и добротностью (Q)
           - Q = f0 / (f2 - f1), где f0 - центральная частота, f1 и f2 - нижняя и верхняя частоты среза
        
        4. **Режекторный фильтр** - не пропускает частоты в определенной полосе.
        
        ## Типичные применения частотного анализа
        
        1. Определение стабильности усилителей
        2. Характеризация фильтров
        3. Анализ цепей обратной связи
        4. Оценка помехоустойчивости
        
        ## Соотношения между параметрами
        
        1. Время нарастания ≈ 0.35 / полоса пропускания (для ФНЧ)
        2. Задержка группы = -dφ/dω, где φ - фаза, ω - угловая частота
        3. Частота среза RC-цепи = 1/(2πRC)
        """
        
        with open("temp_docs/frequency_analysis.txt", "w", encoding="utf-8") as f:
            f.write(frequency_info)
        monte_carlo_info = """
        # Анализ Монте-Карло
        
        Анализ Монте-Карло позволяет оценить влияние разброса параметров компонентов на работу схемы.
        
        ## Основные принципы
        
        1. **Суть метода** - многократное моделирование схемы со случайным разбросом параметров компонентов согласно их допускам.
        
        2. **Распределения параметров**:
           - Равномерное: все значения в диапазоне допуска равновероятны
           - Нормальное (гауссово): бóльшая вероятность значений ближе к номиналу
           - Лог-нормальное: для параметров, которые могут меняться в широком диапазоне
        
        3. **Типичное количество итераций**:
           - Базовая оценка: 50-100 прогонов
           - Детальное исследование: 500-1000 прогонов
           - Критические схемы: 1000+ прогонов
        
        ## Типичные допуски компонентов
        
        1. **Резисторы**:
           - Прецизионные: ±0.1%, ±0.5%, ±1%
           - Стандартные: ±5%, ±10%
        
        2. **Конденсаторы**:
           - Керамические: ±5%, ±10%, ±20%
           - Электролитические: -20%/+80% (или хуже)
           - Пленочные: ±1%, ±5%, ±10%
        
        3. **Полупроводники**:
           - Коэффициент усиления транзисторов: часто ±50% или больше
           - Пороговые напряжения: обычно ±10-20%
        
        ## Анализируемые параметры
        
        1. **Аналоговые схемы**:
           - Коэффициент усиления
           - Частота среза
           - Стабильность
           - Шумовые характеристики
           - Смещение
        
        2. **Цифровые схемы**:
           - Задержки распространения
           - Времена нарастания/спада
           - Параметры фронтов
        
        3. **Смешанные схемы**:
           - Разрешение АЦП/ЦАП
           - Линейность
           - Джиттер
        
        ## Интерпретация результатов
        
        1. **Статистические параметры**:
           - Среднее значение
           - Стандартное отклонение
           - Минимальное и максимальное значения
           - Гистограммы распределения
        
        2. **Критерии качества**:
           - Выход годных (Yield): процент схем, которые соответствуют спецификации
           - Запас по критическим параметрам
        
        3. **Анализ чувствительности**: определение компонентов, которые требуют более жестких допусков
        
        ## Улучшение выхода годных
        
        1. Использование прецизионных компонентов в критичных узлах
        2. Разработка схем с пониженной чувствительностью к разбросу параметров
        3. Калибровка или подстройка в процессе производства
        4. Применение схем автоматической компенсации
        """
        
        with open("temp_docs/monte_carlo.txt", "w", encoding="utf-8") as f:
            f.write(monte_carlo_info)
        
        # Файл с типовыми схемами и их характеристиками (в формате CSV)
        typical_circuits_data = """Circuit Type,Application,Key Parameters,Typical Issues,Optimization Tips
Инвертирующий ОУ,Усиление сигналов,Коэффициент усиления, полоса пропускания,Ограничение скорости нарастания,Выбор ОУ с достаточной скоростью нарастания
Неинвертирующий ОУ,Буферизация и усиление,Входное сопротивление, точность усиления,Смещение нуля при большом Ku,Использование прецизионных ОУ для высоких Ku
Дифференциальный усилитель,Измерительная техника,КОСС, точность усиления,Несогласованность резисторов,Использование прецизионных согласованных резисторов
Фильтр Баттерворта 2-го порядка,Фильтрация сигналов,Частота среза, скорость спада АЧХ,Чувствительность к разбросу параметров,Настройка с помощью переменных резисторов
Полосовой фильтр ГиС,Фильтрация узкой полосы,Добротность, центральная частота,Высокая чувствительность к компонентам,Использование высококачественных индуктивностей
LC-генератор,Генерация синусоидальных сигналов,Стабильность частоты, искажения,Зависимость от температуры,Термокомпенсация и стабилизация амплитуды
Мультивибратор на ОУ,Генерация прямоугольных сигналов,Частота, скважность,Джиттер при низкой частоте,Использование прецизионных компараторов
Малошумящий предусилитель,Усиление слабых сигналов,Коэффициент шума, усиление,Наводки и шумы питания,Тщательное экранирование и фильтрация питания
Источник опорного напряжения,Создание эталонного напряжения,Температурный коэффициент, стабильность,Дрейф с температурой,Выбор компонентов с противоположными ТКН
Импульсный преобразователь,Преобразование напряжения,КПД, пульсации, ЭМИ,Электромагнитные помехи,Правильная компоновка и фильтрация
"""
        
        with open("temp_docs/typical_circuits.csv", "w", encoding="utf-8") as f:
            f.write(typical_circuits_data)
        
        # Создаем базу знаний из временных файлов
        result = build_knowledge_base("temp_docs", "circuit_knowledge_base")
        
        # Удаляем временные файлы
        for file in os.listdir("temp_docs"):
            os.remove(os.path.join("temp_docs", file))
        os.rmdir("temp_docs")
        
        return result
    except Exception as e:
        return {"status": "error", "message": f"Ошибка при создании базы знаний: {str(e)}"}
        
def rag_enhanced_analysis(query, circuit_type=None, kb_path="circuit_knowledge_base"):
    """
    Выполняет поиск в базе знаний для дополнения анализа схемы.
    
    Args:
        query: Запрос пользователя или контекст анализа
        circuit_type: Тип схемы (если известен)
        kb_path: Путь к базе знаний
    
    Returns:
        Дополнительная информация для обогащения ответа
    """
    if not os.path.exists(kb_path):
        create_result = create_sample_circuit_knowledge()
        if create_result["status"] != "success":
            return {
                "status": "error",
                "message": "База знаний не найдена и не может быть создана. Используем только базовые знания модели."
            }
    search_query = query
    if circuit_type:
        search_query = f"{search_query} {circuit_type}"
    analysis_terms = []
    if "стабильность" in query.lower() or "устойчивость" in query.lower():
        analysis_terms.extend(["запас по фазе", "запас по усилению", "стабильность усилителя"])
    if "шум" in query.lower() or "помех" in query.lower():
        analysis_terms.extend(["шумовой анализ", "коэффициент шума", "шумовая температура"])
    if "переход" in query.lower() or "перерегулирование" in query.lower():
        analysis_terms.extend(["переходный процесс", "время нарастания", "перерегулирование"])
    if "частот" in query.lower() or "ачх" in query.lower() or "фчх" in query.lower():
        analysis_terms.extend(["АЧХ", "ФЧХ", "полоса пропускания", "частотный анализ"])
    if "монте" in query.lower() or "разброс" in query.lower() or "допуск" in query.lower():
        analysis_terms.extend(["Монте-Карло", "разброс параметров", "статистический анализ"])
    all_results = []
    for term in analysis_terms:
        results = search_knowledge_base(term, kb_path, top_k=2)
        if results.get("status") == "success" and results.get("results"):
            all_results.extend(results["results"])
    if not all_results:
        results = search_knowledge_base(search_query, kb_path, top_k=3)
        if results.get("status") == "success" and results.get("results"):
            all_results = results["results"]
            
    unique_contents = set()
    unique_results = []
    for result in all_results:
        if result["content"] not in unique_contents:
            unique_contents.add(result["content"])
            unique_results.append(result)
    additional_info = ""
    if unique_results:
        additional_info = "### Дополнительная информация из технической документации:\n\n"
        for i, result in enumerate(unique_results[:5]): 
            content = result["content"].strip()
            source = result.get("source", "техническая документация")
            page = f", стр. {result['page']}" if result.get("page") else ""
            
            additional_info += f"**{i+1}. {source}{page}:**\n{content}\n\n"
    else:
        additional_info = "К сожалению, в базе знаний не найдено дополнительной информации по данному запросу."
    
    return {
        "status": "success",
        "additional_info": additional_info
    }

def combine_analysis_with_rag(analysis_results, query, circuit_name=None):
    """
    Объединяет результаты анализа схемы с дополнительной информацией из RAG.
    
    Args:
        analysis_results: Результаты анализа схемы
        query: Исходный запрос пользователя
        circuit_name: Название схемы (если известно)
    
    Returns:
        Комбинированный отчет
    """
    circuit_type = None
    if circuit_name:
        if "op_amp" in circuit_name.lower() or "operational" in circuit_name.lower():
            circuit_type = "операционный усилитель"
        elif "filter" in circuit_name.lower() or "фильтр" in circuit_name.lower():
            circuit_type = "фильтр"
        elif "power" in circuit_name.lower() or "питание" in circuit_name.lower():
            circuit_type = "источник питания"
        elif "oscillator" in circuit_name.lower() or "генератор" in circuit_name.lower():
            circuit_type = "генератор"
    
    rag_info = rag_enhanced_analysis(query, circuit_type)
    
    if isinstance(analysis_results, dict) and analysis_results.get("summary_table"):
        combined_report = f"## Результаты анализа схемы {circuit_name or ''}\n\n"
        combined_report += analysis_results.get("summary_table", "") + "\n\n"
        combined_report += "## Выводы\n\n"
        if "stability" in analysis_results.get("summary", {}):
            stability_info = analysis_results["summary"]["stability"]
            stable_percent = stability_info.get("stable_percentage", 0)
            if stable_percent >= 90:
                combined_report += "✅ Схема имеет отличную стабильность.\n"
            elif stable_percent >= 70:
                combined_report += "✅ Схема имеет хорошую стабильность, но есть потенциал для улучшения.\n"
            else:
                combined_report += "⚠️ Схема имеет недостаточную стабильность, рекомендуется доработка.\n"
        
        if "frequency" in analysis_results.get("summary", {}):
            freq_info = analysis_results["summary"]["frequency"]
            if freq_info.get("bandwidth"):
                combined_report += f"📊 Полоса пропускания: {freq_info['bandwidth']:.2f} Гц.\n"
            if freq_info.get("max_gain"):
                combined_report += f"📈 Максимальное усиление: {freq_info['max_gain']:.2f} дБ.\n"
        
        if "monte_carlo" in analysis_results.get("summary", {}):
            mc_info = analysis_results["summary"]["monte_carlo"]
            combined_report += "🔄 Результаты анализа Монте-Карло показывают:\n"
            
            if "rise_time" in mc_info:
                rt_info = mc_info["rise_time"]
                combined_report += f"  - Время нарастания: {rt_info['mean']:.2e} ± {rt_info['std']:.2e} с\n"
            
            if "max_gain" in mc_info:
                gain_info = mc_info["max_gain"]
                combined_report += f"  - Усиление: {gain_info['mean']:.2f} ± {gain_info['std']:.2f} дБ\n"
        
        if rag_info.get("status") == "success" and rag_info.get("additional_info"):
            combined_report += "\n" + rag_info["additional_info"]
    else:
        combined_report = f"## Результаты анализа\n\n"
        
        if isinstance(analysis_results, dict):
            for key, value in analysis_results.items():
                combined_report += f"### {key}\n{value}\n\n"
        else:
            combined_report += str(analysis_results) + "\n\n"
        
        if rag_info.get("status") == "success" and rag_info.get("additional_info"):
            combined_report += "\n" + rag_info["additional_info"]
    
    return combined_report