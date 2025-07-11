import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from contextlib import contextmanager
from functools import wraps
from typing import Dict, Optional
import enum

# --- Import aggiuntivi per il web server ---
from flask import Flask
from threading import Thread

# --- Import delle librerie installate ---
import pytz
from dotenv import load_dotenv
import openai
from openai import AsyncOpenAI
from sqlalchemy import (create_engine, Column, Integer, String, Text, DateTime,
                        Enum as SQLAlchemyEnum, func)
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from telegram import (Update, InlineKeyboardButton, InlineKeyboardMarkup, Bot)
from telegram.ext import (Application, CommandHandler, CallbackQueryHandler,
                          MessageHandler, filters, ContextTypes, ConversationHandler)
from dateutil.parser import parse as parse_date

# --- Caricamento iniziale ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

# ==============================================================================
# --- Sezione Web Server per tenere il bot attivo ---
# ==============================================================================
web_app = Flask('')


@web_app.route('/')
def home():
    """Questa rotta serve solo per rispondere al ping del servizio di uptime."""
    return "Il bot è attivo."


def run_web_server():
    """Esegue il server Flask."""
    # L'host 0.0.0.0 è necessario per essere visibile esternamente su Replit
    web_app.run(host='0.0.0.0', port=8080)


def start_web_server_thread():
    """Avvia il server web in un thread separato per non bloccare il bot."""
    server_thread = Thread(target=run_web_server)
    server_thread.daemon = True  # Permette al thread di chiudersi con il programma principale
    server_thread.start()
    logging.info("Web server per l'uptime avviato in background.")


# ==============================================================================
# --- Inizio di: src/config.py ---
# ==============================================================================
class Config:
    """Configurazione centralizzata dell'applicazione"""
    # Bot Configuration
    BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    BOT_NAME = os.getenv('BOT_NAME', 'MoorentPM Bot')

    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    OPENAI_MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS', '1000'))
    OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', '0.7'))

    # Database Configuration
    # Replit usa un file system effimero, quindi usiamo il DB di Replit o un servizio esterno.
    # Per semplicità, Replit offre un DB gratuito che si collega con una URL segreta.
    DATABASE_URL = os.getenv('DATABASE_URL')  # Questa verrà fornita da Replit
    DATABASE_ECHO = os.getenv('DATABASE_ECHO', 'False').lower() == 'true'

    # WhatsApp Configuration
    WHATSAPP_NUMBER = os.getenv('WHATSAPP_NUMBER', '+393534830386')
    WHATSAPP_MESSAGE = os.getenv('WHATSAPP_MESSAGE', 'Ciao! Sono interessato ai servizi MoorentPM')

    # Admin Configuration
    ADMIN_USER_IDS = [int(uid.strip()) for uid in os.getenv('ADMIN_USER_IDS', '').split(',') if uid.strip()]

    # Scheduling Configuration
    TIMEZONE = pytz.timezone(os.getenv('TIMEZONE', 'Europe/Rome'))

    # Linktree URL
    LINKTREE_URL = "https://linktr.ee/moorentpm"

    # Security Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')

    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = 'logs/bot.log'

    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    LOGS_DIR = BASE_DIR / 'logs'

    # Telegram Groups Configuration
    TARGET_GROUPS = [int(gid.strip()) for gid in os.getenv('TARGET_GROUPS', '').split(',') if gid.strip()]

    @classmethod
    def validate(cls):
        """Valida la configurazione all'avvio"""
        errors = []
        if not cls.BOT_TOKEN: errors.append("TELEGRAM_BOT_TOKEN non configurato")
        if not cls.OPENAI_API_KEY: errors.append("OPENAI_API_KEY non configurato")
        if not cls.WHATSAPP_NUMBER: errors.append("WHATSAPP_NUMBER non configurato")
        if not cls.ADMIN_USER_IDS: errors.append("ADMIN_USER_IDS non configurato")
        if not cls.TARGET_GROUPS: errors.append("TARGET_GROUPS non configurato.")
        if not cls.DATABASE_URL: errors.append("DATABASE_URL non configurato (necessario per Replit).")

        if errors:
            raise ValueError("Errori di configurazione:\n" + "\n".join(errors))

        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)
        return True


def setup_logging():
    """Configura il sistema di logging"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO),
        format=log_format,
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('telegram').setLevel(logging.WARNING)
    logging.getLogger('apscheduler').setLevel(logging.WARNING)
    return logging.getLogger(__name__)


# --- Fine di: src/config.py ---


# ==============================================================================
# --- Inizio di: src/models/base.py ---
# ==============================================================================
engine = create_engine(Config.DATABASE_URL, echo=Config.DATABASE_ECHO)
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()


@contextmanager
def get_db():
    """Context manager per gestione sessioni database"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# --- Fine di: src/models/base.py ---


# ==============================================================================
# --- Inizio di: src/models/post.py ---
# ==============================================================================
class PostStatus(enum.Enum):
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    PUBLISHED = "published"
    FAILED = "failed"
    CANCELED = "canceled"


class Post(Base):
    __tablename__ = 'posts'
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    post_type = Column(String(50), nullable=False)
    status = Column(SQLAlchemyEnum(PostStatus), default=PostStatus.DRAFT, nullable=False)
    prompt_used = Column(Text)
    model_version = Column(String(100))
    scheduled_at = Column(DateTime(timezone=True))
    published_at = Column(DateTime(timezone=True))
    telegram_message_id = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(Integer)

    def __repr__(self):
        return f"<Post {self.id}: {self.status.value}>"


# --- Fine di: src/models/post.py ---


# ==============================================================================
# --- Inizio di: src/utils/decorators.py ---
# ==============================================================================
def admin_only(decorated_function):
    """Decoratore per limitare l'accesso ai soli amministratori."""

    @wraps(decorated_function)
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user = update.effective_user
        if not user or user.id not in Config.ADMIN_USER_IDS:
            logging.warning(
                f"Accesso non autorizzato negato a {user.id if user else 'utente sconosciuto'} per {decorated_function.__name__}")
            if update.message:
                await update.message.reply_text("❌ Accesso negato. Questa funzione è riservata agli amministratori.")
            elif update.callback_query:
                await update.callback_query.answer("❌ Accesso negato.", show_alert=True)
            return None
        return await decorated_function(update, context, *args, **kwargs)

    return wrapped


# --- Fine di: src/utils/decorators.py ---


# ==============================================================================
# --- Inizio di: src/services/whatsapp_service.py ---
# ==============================================================================
class WhatsAppService:
    """Gestisce l'integrazione con WhatsApp."""

    @staticmethod
    def generate_url(message: Optional[str] = None) -> str:
        """Genera un URL WhatsApp con un messaggio pre-compilato."""
        phone_number = ''.join(filter(str.isdigit, Config.WHATSAPP_NUMBER))
        if not message:
            message = Config.WHATSAPP_MESSAGE

        params = {'text': message}
        return f"https://wa.me/{phone_number}?{urlencode(params)}"


# --- Fine di: src/services/whatsapp_service.py ---


# ==============================================================================
# --- Inizio di: src/services/ai_service.py ---
# ==============================================================================
class AIService:
    """Servizio per la generazione di contenuti con OpenAI."""

    def __init__(self):
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY non è configurato.")
        self.client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = Config.OPENAI_MODEL

    @staticmethod
    def _get_system_prompt() -> str:
        # Aggiunta istruzione per usare sempre il Linktree corretto
        return f"""Sei un esperto copywriter per MoorentPM, un'azienda leader nella gestione di affitti brevi nel Triveneto. Il tuo stile è professionale ma accogliente, enfatizzando sempre i vantaggi concreti per i proprietari di immobili.
I post devono essere tra 150 e 300 parole.
Mantieni un tono che ispiri fiducia e professionalità.
Alla fine di ogni post, includi SEMPRE una call-to-action che invita a visitare il nostro Linktree: {Config.LINKTREE_URL}"""

    async def generate_post(self, prompt: str) -> Dict:
        """Genera un post utilizzando il modello fine-tuned."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=Config.OPENAI_TEMPERATURE,
                max_tokens=Config.OPENAI_MAX_TOKENS,
            )
            content = response.choices[0].message.content.strip()
            return {'content': content, 'prompt': prompt}
        except openai.APIError as e:
            logging.error(f"Errore API OpenAI: {e}")
            raise
        except Exception as e:
            logging.error(f"Errore generico durante la chiamata a OpenAI: {e}")
            raise

    async def improve_post(self, original_content: str, feedback: str) -> Dict:
        """Migliora un post esistente basandosi sul feedback."""
        prompt = f"""Migliora questo post basandoti sul feedback fornito. Mantieni il messaggio di base ma ottimizza il testo per risolvere i problemi indicati.

POST ORIGINALE:
---
{original_content}
---

FEEDBACK PER LA CORREZIONE:
---
{feedback}
---

Genera la nuova versione migliorata del post."""
        return await self.generate_post(prompt)


# --- Fine di: src/services/ai_service.py ---


# ==============================================================================
# --- Funzione Job per lo Scheduler (spostata qui per evitare PicklingError) ---
# ==============================================================================
async def publish_post_job(post_id: int):
    """Task eseguito dallo scheduler per pubblicare un post."""
    bot = Bot(token=Config.BOT_TOKEN)
    logging.info(f"Esecuzione pubblicazione per post ID: {post_id}")

    post_content = None
    with get_db() as db:
        post = db.get(Post, post_id)
        if not post or post.status != PostStatus.SCHEDULED:
            logging.warning(f"Post {post_id} non trovato o non più in stato SCHEDULED. Annullamento.")
            return
        post_content = post.content

    try:
        # Invia post ai gruppi
        keyboard = [[InlineKeyboardButton("💬 Chatta con noi su WhatsApp", url=WhatsAppService.generate_url())]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        first_message_id = None
        for group_id in Config.TARGET_GROUPS:
            try:
                msg = await bot.send_message(
                    chat_id=group_id,
                    text=post_content,
                    parse_mode='Markdown',
                    reply_markup=reply_markup
                )
                if not first_message_id:
                    first_message_id = msg.message_id
            except Exception as e:
                logging.error(f"Errore durante l'invio al gruppo {group_id}: {e}")

        # Aggiorna il DB in una nuova sessione
        with get_db() as db:
            db.query(Post).filter(Post.id == post_id).update({
                'telegram_message_id': first_message_id,
                'status': PostStatus.PUBLISHED,
                'published_at': datetime.now(Config.TIMEZONE)
            })

        # Notifica admin
        for admin_id in Config.ADMIN_USER_IDS:
            await bot.send_message(chat_id=admin_id, text=f"✅ Post (ID: {post_id}) pubblicato con successo!")

    except Exception as e:
        logging.error(f"Errore nella pubblicazione del post {post_id}: {e}", exc_info=True)
        with get_db() as db:
            db.query(Post).filter(Post.id == post_id).update({'status': PostStatus.FAILED})
        for admin_id in Config.ADMIN_USER_IDS:
            await bot.send_message(chat_id=admin_id, text=f"❌ Errore nella pubblicazione del post (ID: {post_id}): {e}")


# ==============================================================================
# --- Inizio di: src/services/scheduler_service.py ---
# ==============================================================================
class SchedulerService:
    """Gestisce lo scheduling dei post."""

    def __init__(self):
        jobstores = {'default': SQLAlchemyJobStore(url=Config.DATABASE_URL)}
        self.scheduler = AsyncIOScheduler(jobstores=jobstores, timezone=Config.TIMEZONE)

    async def start(self):
        self.scheduler.start()
        logging.info("Scheduler service avviato.")

    async def stop(self):
        if self.scheduler.running:
            self.scheduler.shutdown()
            logging.info("Scheduler service fermato.")

    async def schedule_one_time_post(self, post_id: int, publish_datetime: datetime):
        """Schedula la pubblicazione di un singolo post."""
        job_id = f"one_time_post_{post_id}"
        self.scheduler.add_job(
            publish_post_job,
            'date',
            run_date=publish_datetime,
            args=[post_id],
            id=job_id,
            replace_existing=True
        )
        logging.info(f"Post ID {post_id} schedulato per {publish_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

    async def cancel_scheduled_post(self, post_id: int):
        """Rimuove un job dallo scheduler."""
        job_id = f"one_time_post_{post_id}"
        job = self.scheduler.get_job(job_id)
        if job:
            job.remove()
            logging.info(f"Job rimosso per il post ID {post_id}")
            return True
        return False


# --- Fine di: src/services/scheduler_service.py ---


# ==============================================================================
# --- Inizio di: handlers/user.py ---
# ==============================================================================
async def start_command(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    welcome_message = (
        f"👋 Ciao {user.first_name}!\n\n"
        f"Benvenuto in *{Config.BOT_NAME}*!\n\n"
        "Sono il tuo assistente per la creazione di contenuti. Usa /newpost per iniziare."
    )
    if update.message:
        await update.message.reply_text(welcome_message, parse_mode='Markdown')


async def help_command(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = "📋 *Comandi disponibili:*\n\n/start - Messaggio di benvenuto\n/help - Mostra questo messaggio"
    if update.effective_user and update.effective_user.id in Config.ADMIN_USER_IDS:
        help_text += (
            "\n\n*Comandi Admin:*\n"
            "/newpost - Crea e schedula un nuovo post\n"
            "/scheduled - Vedi e annulla i post schedulati\n"
            "/cancel - Annulla la creazione di un post"
        )
    if update.message:
        await update.message.reply_text(help_text, parse_mode='Markdown')


# --- Fine di: handlers/user.py ---


# ==============================================================================
# --- Inizio di: handlers/admin.py ---
# ==============================================================================
# Stati conversazione
GET_TOPIC, GET_DATETIME, AWAIT_ACTION, GET_CORRECTION = range(4)


@admin_only
async def new_post_entry(update: Update, _: ContextTypes.DEFAULT_TYPE) -> int:
    """Inizia la conversazione per un nuovo post."""
    if update.message:
        await update.message.reply_text(
            "Ciao! Iniziamo a creare un nuovo post.\n\n"
            "Qual è l'argomento o l'idea di base?"
        )
    return GET_TOPIC


async def get_topic(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Salva l'argomento e chiede data/ora."""
    if update.message:
        context.user_data['topic'] = update.message.text
        await update.message.reply_text(
            "Ottimo. Ora dimmi **quando** vuoi pubblicarlo.\n"
            "Puoi usare un formato naturale (es. `domani alle 15:00`) o specifico (`25/12/2025 10:30`)."
        )
    return GET_DATETIME


async def get_datetime_and_generate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Salva data/ora, genera il post e mostra l'anteprima."""
    if not update.message:
        return GET_DATETIME

    datetime_str = update.message.text
    try:
        publish_dt = parse_date(datetime_str, dayfirst=True)
        publish_dt = Config.TIMEZONE.localize(publish_dt)

        if publish_dt < datetime.now(Config.TIMEZONE):
            await update.message.reply_text("⚠️ La data inserita è nel passato. Per favore, inserisci una data futura.")
            return GET_DATETIME

        context.user_data['publish_dt'] = publish_dt
    except ValueError:
        await update.message.reply_text("Non ho capito la data. Prova un formato come `GG/MM/AAAA HH:MM`.")
        return GET_DATETIME

    await update.message.reply_text("Perfetto. Sto generando il post con l'AI... 🤖")

    topic = context.user_data.get('topic', 'un argomento di marketing')
    prompt = f"Crea un post per social media per MoorentPM. L'argomento è: '{topic}'."

    ai_service = AIService()
    try:
        post_data = await ai_service.generate_post(prompt)
        context.user_data['post_content'] = post_data['content']
        context.user_data['prompt'] = post_data['prompt']

        with get_db() as db:
            new_post = Post(
                content=post_data['content'],
                prompt_used=post_data['prompt'],
                post_type='custom',
                model_version=ai_service.model,
                status=PostStatus.DRAFT,
                created_by=update.effective_user.id
            )
            db.add(new_post)
            db.commit()
            context.user_data['post_id'] = new_post.id

        await show_preview_and_actions(update, context)
        return AWAIT_ACTION

    except Exception as e:
        logging.error(f"Errore durante la generazione del post: {e}")
        await update.message.reply_text(f"❌ Si è verificato un errore durante la generazione: {e}")
        return ConversationHandler.END


async def show_preview_and_actions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Mostra l'anteprima del post e i bottoni di azione."""
    post_content = context.user_data.get('post_content', 'Contenuto non disponibile.')
    publish_dt = context.user_data.get('publish_dt')
    publish_dt_str = publish_dt.strftime('%d/%m/%Y alle %H:%M') if publish_dt else 'N/D'

    text = (
        f"✨ *ANTEPRIMA POST*\n"
        f"*(Schedulato per: {publish_dt_str})*\n\n"
        f"---\n\n"
        f"{post_content}\n\n"
        f"---\n\n"
        "Cosa vuoi fare?"
    )

    keyboard = [
        [InlineKeyboardButton("✅ Schedula", callback_data="post_schedule")],
        [InlineKeyboardButton("✏️ Correggi", callback_data="post_correct")],
        [InlineKeyboardButton("🗑 Scarta", callback_data="post_discard")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    if update.callback_query:
        await update.callback_query.edit_message_text(text, parse_mode='Markdown', reply_markup=reply_markup)
    elif update.message:
        await update.message.reply_text(text, parse_mode='Markdown', reply_markup=reply_markup)


async def await_action(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Gestisce i bottoni di azione (Schedula, Correggi, Scarta)."""
    query = update.callback_query
    if not query:
        return AWAIT_ACTION

    await query.answer()

    action = query.data
    post_id = context.user_data.get('post_id')
    if not post_id:
        await query.edit_message_text("❌ Errore: non trovo più il post a cui ti riferisci. Annulla e ricomincia.")
        return ConversationHandler.END

    if action == "post_schedule":
        publish_dt = context.user_data['publish_dt']

        with get_db() as db:
            db.query(Post).filter(Post.id == post_id).update({
                'status': PostStatus.SCHEDULED,
                'scheduled_at': publish_dt
            })

        scheduler_service = context.application.bot_data.get('scheduler_service')
        await scheduler_service.schedule_one_time_post(post_id, publish_dt)

        await query.edit_message_text(
            f"✅ Perfetto! Post schedulato per il {publish_dt.strftime('%d/%m/%Y alle %H:%M')}."
        )
        return ConversationHandler.END

    elif action == "post_correct":
        await query.edit_message_text(
            "Ok, dimmi come devo correggerlo (es. 'rendilo più formale', 'aggiungi un aneddoto', etc.).")
        return GET_CORRECTION

    elif action == "post_discard":
        with get_db() as db:
            db.query(Post).filter(Post.id == post_id).delete()

        await query.edit_message_text("🗑 Post scartato.")
        return ConversationHandler.END

    return AWAIT_ACTION


async def get_correction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Riceve il feedback, rigenera il post e mostra la nuova anteprima."""
    if not update.message:
        return GET_CORRECTION

    feedback = update.message.text
    original_content = context.user_data['post_content']

    await update.message.reply_text("Sto applicando le tue correzioni... 🤖")

    ai_service = AIService()
    try:
        improved_data = await ai_service.improve_post(original_content, feedback)
        new_content = improved_data['content']

        context.user_data['post_content'] = new_content
        context.user_data['prompt'] = improved_data['prompt']
        post_id = context.user_data['post_id']

        with get_db() as db:
            db.query(Post).filter(Post.id == post_id).update({
                'content': new_content,
                'prompt_used': improved_data['prompt']
            })

        await show_preview_and_actions(update, context)
        return AWAIT_ACTION

    except Exception as e:
        logging.error(f"Errore durante la correzione del post: {e}")
        await update.message.reply_text(f"❌ Si è verificato un errore: {e}")
        return ConversationHandler.END


async def cancel_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Annulla la conversazione in corso."""
    if 'post_id' in context.user_data:
        with get_db() as db:
            db.query(Post).filter(Post.id == context.user_data['post_id'], Post.status == PostStatus.DRAFT).delete()

    if update.message:
        await update.message.reply_text("Operazione annullata.")
    context.user_data.clear()
    return ConversationHandler.END


@admin_only
async def list_scheduled_posts(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    """Mostra una lista dei post schedulati e permette di annullarli."""
    message_sender = update.message or (update.callback_query and update.callback_query.message)
    if not message_sender:
        return

    with get_db() as db:
        scheduled_posts = db.query(Post).filter(Post.status == PostStatus.SCHEDULED).order_by(Post.scheduled_at).all()

        if not scheduled_posts:
            text = "📭 Non ci sono post schedulati al momento."
            if update.callback_query:
                await update.callback_query.edit_message_text(text)
            else:
                await message_sender.reply_text(text)
            return

        message = "🗓️ *Post Schedulati:*\n\n"
        keyboard = []
        for post in scheduled_posts:
            schedule_time = post.scheduled_at.strftime('%d/%m/%Y %H:%M')
            message += f"▪️ *ID:* {post.id} | *Per il:* {schedule_time}\n`{post.content[:70]}...`\n\n"
            keyboard.append(
                [InlineKeyboardButton(f"🗑 Annulla Post ID {post.id}", callback_data=f"cancel_post_{post.id}")])

    reply_markup = InlineKeyboardMarkup(keyboard)

    if update.callback_query:
        await update.callback_query.edit_message_text(message, parse_mode='Markdown', reply_markup=reply_markup)
    else:
        await message_sender.reply_text(message, parse_mode='Markdown', reply_markup=reply_markup)


async def cancel_scheduled_post_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Callback per annullare un post schedulato."""
    query = update.callback_query
    await query.answer()

    post_id = int(query.data.split('_')[-1])

    with get_db() as db:
        db.query(Post).filter(Post.id == post_id, Post.status == PostStatus.SCHEDULED).update({
            'status': PostStatus.CANCELED
        })

    scheduler_service = context.application.bot_data.get('scheduler_service')
    await scheduler_service.cancel_scheduled_post(post_id)

    await query.answer("✅ Schedulazione annullata.")
    await list_scheduled_posts(update, context)


# --- Fine di: handlers/admin.py ---


# ==============================================================================
# --- Inizio di: bot.py (classe MoorentBot) e run.py (entry point) ---
# ==============================================================================
class MoorentBot:
    """Classe principale del bot Telegram."""

    def __init__(self):
        try:
            Config.validate()
            self.logger = setup_logging()
        except ValueError as e:
            logging.basicConfig(level=logging.ERROR)
            logging.error(f"Errore di configurazione critico: {e}")
            sys.exit(1)

        self.app = Application.builder().token(Config.BOT_TOKEN).build()
        self.scheduler_service = None

    async def post_init(self, application: Application) -> None:
        """Inizializzazione post-avvio del bot."""
        self.logger.info(f"Bot {Config.BOT_NAME} avviato con successo!")

        Base.metadata.create_all(bind=engine)
        self.logger.info("Database inizializzato.")

        self.scheduler_service = SchedulerService()
        application.bot_data['scheduler_service'] = self.scheduler_service
        await self.scheduler_service.start()

        for admin_id in Config.ADMIN_USER_IDS:
            try:
                await application.bot.send_message(
                    chat_id=admin_id,
                    text=f"✅ {Config.BOT_NAME} è online!"
                )
            except Exception as e:
                self.logger.error(f"Errore notifica admin {admin_id}: {e}")

    async def post_shutdown(self, _: Application) -> None:
        """Operazioni di pulizia durante lo spegnimento."""
        self.logger.info("Spegnimento del bot in corso...")
        if self.scheduler_service:
            await self.scheduler_service.stop()

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Gestione centralizzata degli errori."""
        self.logger.error(f"Eccezione durante la gestione di un update: {context.error}", exc_info=context.error)

        if isinstance(update, Update) and hasattr(update, 'effective_message') and update.effective_message:
            try:
                await update.effective_message.reply_text("❌ Si è verificato un errore. Il team è stato notificato.")
            except Exception:
                pass

    def setup_handlers(self) -> None:
        """Registra tutti gli handler del bot."""

        new_post_conv = ConversationHandler(
            entry_points=[CommandHandler("newpost", new_post_entry)],
            states={
                GET_TOPIC: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_topic)],
                GET_DATETIME: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_datetime_and_generate)],
                AWAIT_ACTION: [CallbackQueryHandler(await_action, pattern="^post_")],
                GET_CORRECTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_correction)],
            },
            fallbacks=[CommandHandler("cancel", cancel_conversation)],
        )

        self.app.add_handler(CommandHandler("start", start_command))
        self.app.add_handler(CommandHandler("help", help_command))
        self.app.add_handler(new_post_conv)
        self.app.add_handler(CommandHandler("scheduled", list_scheduled_posts))
        self.app.add_handler(CallbackQueryHandler(cancel_scheduled_post_callback, pattern="^cancel_post_"))

        async def handle_generic_text(update: Update, _: ContextTypes.DEFAULT_TYPE):
            if update.effective_user and update.effective_user.id in Config.ADMIN_USER_IDS:
                if update.message:
                    await update.message.reply_text(
                        "Ciao Admin! Usa /newpost per creare e schedulare un post."
                    )
            else:
                if update.message:
                    await update.message.reply_text("Usa /start per iniziare o /help per vedere i comandi.")

        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_generic_text))
        self.app.add_error_handler(self.error_handler)
        self.logger.info("Handler registrati con successo.")

    def run(self):
        """Avvia il bot."""
        self.setup_handlers()
        self.app.post_init = self.post_init
        self.app.post_shutdown = self.post_shutdown

        self.logger.info(f"Avvio di {Config.BOT_NAME}...")
        self.app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    # Avvia il server web in un thread separato per tenere vivo il Repl
    start_web_server_thread()

    # Avvia il bot
    bot = MoorentBot()
    bot.run()