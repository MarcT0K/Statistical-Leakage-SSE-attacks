"""Everything needed to extract Enron and Apache email datasets.
"""

import email
import functools
import glob
import mailbox
import pickle
import os

import colorlog
import pandas as pd
import tqdm

from bs4 import BeautifulSoup

from .keyword_extract import KeywordExtractor

logger = colorlog.getLogger("RaaC paper")


def split_df(dframe, frac=0.5):
    first_split = dframe.sample(frac=frac)
    second_split = dframe.drop(first_split.index)
    return first_split, second_split


def get_body_from_enron_email(mail):
    """Extract the content from raw Enron email"""
    msg = email.message_from_string(mail)
    parts = []
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            parts.append(part.get_payload())
    return "".join(parts)


def get_body_from_mboxmsg(msg):
    """Extract the content from a raw Apache email"""
    parts = []
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            parts.append(part.get_payload())
    body = "".join(parts)
    # Remove the mailing list signature
    body = body.split("To unsubscribe", maxsplit=1)[0]
    return body


def extract_enron_sent_emails(maildir_directory="./maildir/") -> pd.DataFrame:
    """Extract the emails from the _sent_mail folder of each Enron mailbox."""
    path = os.path.expanduser(maildir_directory)
    mails = glob.glob(f"{path}/*/_sent_mail/*")

    mail_contents = []
    for mailfile_path in tqdm.tqdm(iterable=mails, desc="Reading the emails"):
        with open(mailfile_path, "r", encoding="utf-8") as mailfile:
            raw_mail = mailfile.read()
            mail_contents.append(get_body_from_enron_email(raw_mail))

    return pd.DataFrame(data={"filename": mails, "mail_body": mail_contents})


def extract_apache_ml(maildir_directory="./apache_ml/") -> pd.DataFrame:
    """Extract all the emails sent on the Apache Lucene mailing list between 2002 and 2011."""
    path = os.path.expanduser(maildir_directory)
    mails = glob.glob(f"{path}/*")
    mail_contents = []
    mail_ids = []
    for mbox_path in tqdm.tqdm(iterable=mails, desc="Reading the emails"):
        for mail in mailbox.mbox(mbox_path):
            mail_content = get_body_from_mboxmsg(mail)
            mail_contents.append(mail_content)
            mail_ids.append(mail["Message-ID"])
    return pd.DataFrame(data={"filename": mail_ids, "mail_body": mail_contents})


def extract_apache_ml_by_year(
    from_year=2002, to_year=2012, maildir_directory="./apache_ml/"
) -> pd.DataFrame:
    path = os.path.expanduser(maildir_directory)
    mail_contents = []
    mail_ids = []

    for year in range(from_year, to_year):
        mails = glob.glob(f"{path}/{year}*")
        for mbox_path in tqdm.tqdm(iterable=mails, desc="Reading the emails"):
            for mail in mailbox.mbox(mbox_path):
                mail_content = get_body_from_mboxmsg(mail)
                mail_contents.append(mail_content)
                mail_ids.append(mail["Message-ID"])
    return pd.DataFrame(data={"filename": mail_ids, "mail_body": mail_contents})


def extract_blogs(blog_dir="./blogs") -> pd.DataFrame:
    path = os.path.expanduser(blog_dir)
    blogs = glob.glob(f"{path}/*")
    posts = []
    post_ids = []

    for blog_path in tqdm.tqdm(iterable=blogs, desc="Reading the blogs"):
        with open(blog_path, "r", errors="ignore", encoding="utf-8") as blog_file:
            blog = blog_file.read()

        soup = BeautifulSoup(blog, "xml")
        i = 1
        for post in soup.find_all("post"):
            posts.append(str(post.string))
            post_ids.append(f"{blog_file.name}_{i}")
            i += 1

    return pd.DataFrame(data={"filename": post_ids, "mail_body": posts})


def generic_extractor(extract_function, dataset_name, voc_size):
    file_name = f"{dataset_name}_extractor_{voc_size}.pkl"
    if not os.path.isfile(file_name):
        docs = extract_function()
        extractor = KeywordExtractor(docs, voc_size)
        with open(file_name, "wb") as pkl_file:
            pickle.dump(extractor, pkl_file)
    else:
        with open(file_name, "rb") as pkl_file:
            extractor = pickle.load(pkl_file)
    return extractor


enron_extractor = functools.partial(
    generic_extractor, extract_enron_sent_emails, "enron"
)

apache_extractor = functools.partial(generic_extractor, extract_apache_ml, "apache")

blogs_extractor = functools.partial(generic_extractor, extract_blogs, "blogs")

apache_by_year_extractor = functools.partial(
    generic_extractor, extract_apache_ml_by_year, "apache_by_year"
)
