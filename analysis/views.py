import os

import requests
from django.http import JsonResponse
from django.shortcuts import render, redirect
import pandas as pd
from googleapiclient.discovery import build
from concurrent.futures import ThreadPoolExecutor
from .analyze import compute


API_KEY = os.getenv("API_KEY", "")


def get_client_ip(request):
    # Check for X-Forwarded-For header if Django is behind a proxy
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')

    if x_forwarded_for:
        # X-Forwarded-For can return a list of IPs if there are multiple proxies,
        # so we take the first one (the original client IP)
        ip = x_forwarded_for.split(',')[0]
    else:
        # Use REMOTE_ADDR if no proxy is being used
        ip = request.META.get('REMOTE_ADDR')

    return ip


def get_comment_count(video_id):
    # Build the YouTube service
    youtube = build('youtube', 'v3', developerKey=API_KEY)

    # Get video details (including comment count)
    request = youtube.videos().list(
        part='statistics',
        id=video_id
    )
    response = request.execute()

    # Extract the comment count
    if 'items' in response and len(response['items']) > 0:
        comment_count = response['items'][0]['statistics'].get('commentCount', '0')
        return int(comment_count)
    else:
        return "Video not found or no comments available."


def fetch_comments(video_id):
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    comments = []
    next_page_token = None
    print(f"{video_id}: Creating Dataframe")

    def fetch_page(page_token=None):
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=page_token
        )
        return request.execute()

    # Fetch all pages using threading to speed up the process
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        while True:
            response = fetch_page(next_page_token)
            futures.append(response)

            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)

            next_page_token = response.get('nextPageToken')
            if next_page_token is None:
                break
    print(f"{video_id}: DataFrame Created")
    return pd.DataFrame(comments, columns=['comments'])


def youtube_input(request):
    if request.method == "POST":
        print(get_client_ip(request))
        video_url = request.POST.get('video_url')
        video_id = video_url.split('v=')[-1].split("&t=")[0].split("?feature=")[0].split("&ab_channel=")[0].split("&list=")[0]

        # Store the video_id in session
        request.session['video_id'] = video_id

        # Redirect to the loader page
        return redirect('loading_page')

    return render(request, 'input.html')


def loader_page(request):
    # Retrieve video_id from session
    video_id = request.session.get('video_id')

    if not video_id:
        return redirect('youtube_input')  # If no video ID, redirect to input page

    # Fetch the comments from YouTube
    total_comments = get_comment_count(video_id)
    request.session["total_comments"] = total_comments

    # Count total comments to influence the progress bar timing
    # total_comments = len(comments_df)

    # Render the loader page and pass the total number of comments
    return render(request, 'loading.html', {"video_id": video_id, "total_comments": total_comments})



def process_data(request):
    # Retrieve video_id from session
    video_id = request.session.get('video_id')

    if not video_id:
        return JsonResponse({'error': 'No video ID found'}, status=400)

    # Fetch the comments from YouTube
    comments_df = fetch_comments(video_id)

    # Run the model predictions (positivity, spam, violence)
    df = compute(comments_df, video_id)

    # Calculate additional fields
    total_comments = len(df)
    positive_comments = len(df[df["positivity"] == 1])
    negative_comments = len(df[df["positivity"] == -1])
    neutral_comments = len(df[df["positivity"] == 0])
    spam_comments = len(df[df["spam"] == 1])
    violent_comments = len(df[df["violence"] == 1])

    # Get the top language
    top_language = df['language'].value_counts().idxmax()
    predominant_sentiment = "positive" if positive_comments > negative_comments else "negative"
    predominant_sentiment = predominant_sentiment if max(positive_comments, negative_comments) > neutral_comments and positive_comments != negative_comments else "neutral"


    # Store the results in session
    request.session['analysis_results'] = {
        "actual_total": request.session.get("total_comments"),
        "percentage": round((total_comments / request.session.get("total_comments")) * 100, 1),
        "total_comments": total_comments,
        "positive_comments": positive_comments,
        "negative_comments": negative_comments,
        "neutral_comments": neutral_comments,
        "spam_comments": spam_comments,
        "violent_comments": violent_comments,
        "top_language": top_language,
        "spam_percentage": round((spam_comments / total_comments) * 100, 1),
        "violent_percentage": round((violent_comments / total_comments) * 100, 1),
        "predominant_sentiment": predominant_sentiment
    }

    # Return JSON response once processing is done
    return JsonResponse({'success': True})


def results(request):
    # Retrieve the results from session
    analysis_results = request.session.get('analysis_results')

    if not analysis_results:
        return redirect('youtube_input')  # If no results, redirect to input page

    # Pass the results to the template
    return render(request, 'result.html', analysis_results)
